#!/usr/bin/env node
/** Build only the public-safe static gallery, with evidence and artifact checks. */
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
const root=process.cwd(), source=path.join(root,'site');
const outIndex=process.argv.indexOf('--out');
if(outIndex>=0&&!process.argv[outIndex+1])throw new Error('--out requires a directory');
const output=path.resolve(root,outIndex<0?'temp/pages-dist':process.argv[outIndex+1]);
if(output===root||!output.startsWith(root+path.sep)||output===source||output.startsWith(source+path.sep))throw new Error('Output must be a separate contained workspace directory');
const data=JSON.parse(await fs.readFile(path.join(source,'data.json'),'utf8'));
const manifest=JSON.parse(await fs.readFile(path.join(source,'assets.json'),'utf8'));
const allowed=new Set(['index.html','styles.css','app.js','data.json','assets.json','.nojekyll']);
const privateText=/(?:\b[A-Za-z]:[\\/]|\/Users\/|\/home\/|AppData[\\/]|\.codex|\bsneak\b|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|AKIA[0-9A-Z]{16})/i;
if(data.schema!==1||data.families.length!==12||new Set(data.families.map(x=>x.id)).size!==12)throw new Error('Expected twelve unique frozen families');
if(data.runtime.blender!=='5.2.2 LTS'||data.evidenceDate!=='2026-10-08')throw new Error('Unsupported evidence snapshot');
const actual=data.families.filter(x=>x.output);
if(actual.length!==2||actual.some(x=>x.accepted!==false))throw new Error('Actual acceptance must retain its required blockers');
for(const family of data.families){
 if(!/^[a-z_]+$/.test(family.id))throw new Error('Invalid family identifier');
 if(!family.output&&family.status!=='reference')throw new Error('Reference cannot be presented as a reconstructed result');
 if(family.metrics)for(const metric of Object.values(family.metrics)){
  const pass=metric.area_iou>=data.silhouetteLimits.area&&metric.boundary_iou>=data.silhouetteLimits.boundary&&metric.signed_distance_loss<=data.silhouetteLimits.distance;
  if(pass!==metric.passed)throw new Error('A per-view verdict contradicts its unchanged limits');
 }
 if(family.id==='rounded_triangle_dot'&&(family.surface.limits||family.verdicts.surface!=='unqualified'))throw new Error('Triangle surface qualification is not established');
}
if(data.negativeControls.length!==3||!data.negativeControls.every(x=>x.caught))throw new Error('Missing actual negative-control evidence');
const listed=new Set(manifest.assets.map(x=>x.path));
let total=0,files=[];
async function inventory(dir){for(const item of await fs.readdir(dir,{withFileTypes:true})){const absolute=path.join(dir,item.name),relative=path.relative(source,absolute).split(path.sep).join('/');if(item.isSymbolicLink())throw new Error('Links cannot enter a Pages artifact');if(item.isDirectory()){if(relative!=='assets')throw new Error('Unexpected site directory');await inventory(absolute);}else if(item.isFile()){if(!allowed.has(relative)&&!listed.has(relative))throw new Error('Unlisted public file '+relative);const bytes=await fs.readFile(absolute);total+=bytes.length;if(relative.endsWith('.png')){const record=manifest.assets.find(x=>x.path===relative);if(!record||createHash('sha256').update(bytes).digest('hex')!==record.sha256||bytes.length!==record.bytes)throw new Error('Asset identity changed '+relative);let offset=8;while(offset<bytes.length){const size=bytes.readUInt32BE(offset),kind=bytes.subarray(offset+4,offset+8).toString('ascii');if(['tEXt','zTXt','iTXt','eXIf'].includes(kind))throw new Error('PNG ancillary metadata must be reviewed');offset+=size+12;}}else if(privateText.test(bytes.toString('utf8')))throw new Error('Private machine path or credential pattern in '+relative);files.push(relative);}else throw new Error('Only regular files may be published');}}
await inventory(source);
if(total>15*1024*1024||manifest.assets.length!==listed.size||manifest.assets.some(x=>!files.includes(x.path)))throw new Error('Asset inventory incomplete or exceeds bounded publication size');
const assetsUsed=[];function collect(value){if(typeof value==='string'&&value.startsWith('assets/'))assetsUsed.push(value);else if(Array.isArray(value))value.forEach(collect);else if(value&&typeof value==='object')Object.values(value).forEach(collect);}collect(data);
for(const used of assetsUsed)if(!listed.has(used))throw new Error('Missing referenced image '+used);
const html=await fs.readFile(path.join(source,'index.html'),'utf8');
const ids=new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]));
for(const match of html.matchAll(/(?:href|src)="([^"]+)"/g)){const value=match[1];if(value.startsWith('#')){if(value!=='#'&&!ids.has(value.slice(1)))throw new Error('Broken fragment '+value);}else if(!value.startsWith('https://')&&!files.includes(value)&&value!=='build.json')throw new Error('Missing HTML link '+value);}
let commit=process.env.BUILD_SHA||execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim();if(!/^[a-f0-9]{40}$/.test(commit))throw new Error('Build source SHA is invalid');
await fs.mkdir(path.dirname(output),{recursive:true});await fs.mkdir(output);
await fs.cp(source,output,{recursive:true,errorOnExist:true});
const build={commit,evidenceDate:data.evidenceDate,evidenceSource:data.sourceSnapshot,blender:data.runtime.blender,assetCount:listed.size,bytes:total,builtAt:new Date().toISOString()};
await fs.writeFile(path.join(output,'build.json'),JSON.stringify(build,null,2)+'\n');
console.log(JSON.stringify({status:'built',...build},null,2));
