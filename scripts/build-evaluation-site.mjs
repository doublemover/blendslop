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
const allowed=new Set(['index.html','styles.css','app.js','data.json','diagnostics.json','assets.json','.nojekyll']);
// Optional local continuation stays additive to the frozen historical gallery.
let continuation=null;
try{continuation=JSON.parse(await fs.readFile(path.join(source,'continuation-assets.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
const continuationAssets=continuation?[...continuation.images,...continuation.measurement_illustrations]:[];
if(continuation){
 if(continuation.protocol!=='quality_continuation_preview_assets_v1'||continuation.local_only!==true||continuationAssets.length!==4)throw new Error('Unexpected continuation evidence');
 for(const record of continuationAssets){
  if(!/^assets\/[a-z_]+-continuation-[a-z]+\.png$/.test(record.path)||!/^docs\/[a-z0-9/_-]+\.png$/.test(record.source)||!Number.isInteger(record.bytes)||record.bytes<=0||record.pixel_payload_preserved!==true||![record.sha256,record.original_sha256,record.idat_sha256].every(x=>/^[a-f0-9]{64}$/.test(x)))throw new Error('Invalid continuation asset binding');
 }
 for(const file of ['continuation.html','continuation.css','continuation-assets.json'])allowed.add(file);
}
const allAssets=[...manifest.assets,...continuationAssets];
const assetRecords=new Map(allAssets.map(record=>[record.path,record]));
if(assetRecords.size!==allAssets.length)throw new Error('Duplicate public asset identity');
const privateText=/(?:\b[A-Za-z]:[\\/]|\/Users\/|\/home\/|AppData[\\/]|\.codex|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|AKIA[0-9A-Z]{16})/i;
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
const diagnostics=JSON.parse(await fs.readFile(path.join(source,'diagnostics.json'),'utf8'));
if(diagnostics.schema!==1||Object.keys(diagnostics.cases).length!==2)throw new Error('Missing exact diagnostic evidence');
function countRuns(runs,width,height){let count=0,lastY=-1,lastEnd=0;for(const run of runs){if(run.length!==3||!run.every(Number.isInteger))throw new Error('Invalid diagnostic run');const[y,x0,x1]=run;if(y<0||y>=height||x0<0||x1>width||x0>=x1||y<lastY||(y===lastY&&x0<lastEnd))throw new Error('Diagnostic runs overlap or leave image');count+=x1-x0;lastY=y;lastEnd=x1;}return count;}
for(const family of actual){
 if(!family.inspection?.oblique_35_28||family.thumbnail!==family.inspection.oblique_35_28)throw new Error('Actual families must default to real shaded output');
 if(family.preview&&family.preview.geometryHash!==family.geometry)throw new Error('Preview changed measured geometry');
 for(const view of data.views){const row=diagnostics.cases[family.id]?.[view],metric=family.metrics[view];if(!row||row.width!==512||row.height!==512)throw new Error('Missing diagnostic view');
  for(const role of ['reference','output']){const asset=manifest.assets.find(x=>x.path===row[role]);if(row[role]!==family[role][view]||asset?.sha256!==row[role+'Sha256'])throw new Error('Diagnostic source identity mismatch');}
  for(const [runs,count] of [['missingRuns','missingPixels'],['extraRuns','extraPixels'],['missingBandRuns','missingBandPixels'],['extraBandRuns','extraBandPixels']])if(countRuns(row[runs],row.width,row.height)!==row[count])throw new Error('Diagnostic count mismatch');
  if(row.areaUnion-row.areaIntersection!==row.missingPixels+row.extraPixels||row.bandUnion-row.bandIntersection!==row.missingBandPixels+row.extraBandPixels||Math.abs(row.areaIntersection/row.areaUnion-metric.area_iou)>1e-12||Math.abs(row.bandIntersection/row.bandUnion-metric.boundary_iou)>1e-12)throw new Error('Diagnostic overlay contradicts exact metrics');
 }
}
if(data.negativeControls.length!==3||!data.negativeControls.every(x=>x.caught))throw new Error('Missing actual negative-control evidence');
const listed=new Set(allAssets.map(x=>x.path));
let total=0,files=[];
async function inventory(dir){for(const item of await fs.readdir(dir,{withFileTypes:true})){const absolute=path.join(dir,item.name),relative=path.relative(source,absolute).split(path.sep).join('/');if(item.isSymbolicLink())throw new Error('Links cannot enter a Pages artifact');if(item.isDirectory()){if(relative!=='assets')throw new Error('Unexpected site directory');await inventory(absolute);}else if(item.isFile()){if(!allowed.has(relative)&&!listed.has(relative))throw new Error('Unlisted public file '+relative);const bytes=await fs.readFile(absolute);total+=bytes.length;if(relative.endsWith('.png')){const record=assetRecords.get(relative);if(!record||createHash('sha256').update(bytes).digest('hex')!==record.sha256||bytes.length!==record.bytes)throw new Error('Asset identity changed '+relative);if(bytes.subarray(0,8).toString('hex')!=='89504e470d0a1a0a')throw new Error('Invalid PNG signature');let offset=8,idat=[];while(offset<bytes.length){const size=bytes.readUInt32BE(offset),kind=bytes.subarray(offset+4,offset+8).toString('ascii');if(offset+size+12>bytes.length)throw new Error('Truncated PNG');if(['tEXt','zTXt','iTXt','eXIf'].includes(kind))throw new Error('PNG ancillary metadata must be reviewed');if(kind==='IDAT')idat.push(bytes.subarray(offset+8,offset+8+size));offset+=size+12;}if(offset!==bytes.length)throw new Error('Invalid PNG length');if(record.idat_sha256&&(createHash('sha256').update(Buffer.concat(idat)).digest('hex')!==record.idat_sha256||bytes.readUInt32BE(16)!==record.width||bytes.readUInt32BE(20)!==record.height))throw new Error('Continuation pixel payload changed');}else if(privateText.test(bytes.toString('utf8'))||/[\u00c2\u00c3]/.test(bytes.toString('utf8')))throw new Error('Private machine path or credential pattern in '+relative);files.push(relative);}else throw new Error('Only regular files may be published');}}
await inventory(source);
if(total>15*1024*1024||allAssets.length!==listed.size||allAssets.some(x=>!files.includes(x.path)))throw new Error('Asset inventory incomplete or exceeds bounded publication size');
const assetsUsed=[];function collect(value){if(typeof value==='string'&&value.startsWith('assets/'))assetsUsed.push(value);else if(Array.isArray(value))value.forEach(collect);else if(value&&typeof value==='object')Object.values(value).forEach(collect);}collect(data);
for(const used of assetsUsed)if(!listed.has(used))throw new Error('Missing referenced image '+used);
for(const page of continuation?['index.html','continuation.html']:['index.html']){
 const html=await fs.readFile(path.join(source,page),'utf8');
 const idList=[...html.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]),ids=new Set(idList);
 if(ids.size!==idList.length)throw new Error('Duplicate HTML fragment in '+page);
 for(const match of html.matchAll(/(?:href|src)="([^"]+)"/g)){const value=match[1];if(value.startsWith('#')){if(value!=='#'&&!ids.has(value.slice(1)))throw new Error('Broken fragment '+value);}else if(!value.startsWith('https://')&&!files.includes(value)&&value!=='build.json')throw new Error('Missing HTML link '+value);}
}
let commit=process.env.BUILD_SHA||execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim();if(!/^[a-f0-9]{40}$/.test(commit))throw new Error('Build source SHA is invalid');
await fs.mkdir(path.dirname(output),{recursive:true});await fs.mkdir(output);
await fs.cp(source,output,{recursive:true,errorOnExist:true});
const build={commit,evidenceDate:data.evidenceDate,evidenceSource:data.sourceSnapshot,blender:data.runtime.blender,assetCount:listed.size,continuationAssetCount:continuationAssets.length,bytes:total,builtAt:new Date().toISOString()};
await fs.writeFile(path.join(output,'build.json'),JSON.stringify(build,null,2)+'\n');
console.log(JSON.stringify({status:'built',...build},null,2));
