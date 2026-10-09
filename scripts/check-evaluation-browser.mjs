#!/usr/bin/env node
/** Bounded hosted-runner browser check; uses existing Chrome, no install. */
import http from 'node:http';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
function argument(name,fallback){const index=process.argv.indexOf(name);if(index<0)return fallback;if(!process.argv[index+1])throw new Error(name+' requires a directory');return process.argv[index+1];}
const directory=path.resolve(argument('--dir','temp/pages-dist')), evidence=path.resolve(argument('--out','temp/pages-browser'));
for(const value of [directory,evidence])if(!value.startsWith(process.cwd()+path.sep))throw new Error('Browser check paths must stay inside the workspace');
let current=null;try{current=JSON.parse(await fs.readFile(path.join(directory,'latest.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
await fs.mkdir(evidence,{recursive:true});
const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.png':'image/png'};
const server=http.createServer(async(req,res)=>{try{const file=path.resolve(directory,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname==='/'?'/index.html':new URL(req.url,'http://localhost').pathname));if(!file.startsWith(directory+path.sep))throw new Error('outside site');const bytes=await fs.readFile(file);res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');res.end(bytes);}catch{res.statusCode=404;res.end('Not found');}});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const profile=await fs.mkdtemp(path.join(evidence,'chrome-'));
const chrome=spawn(process.env.CHROME_BINARY||'google-chrome',['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:['ignore','ignore','pipe']});
let socket, session, sequence=0, pending=new Map(), errors=[];
const timer=setTimeout(()=>{for(const item of pending.values())item.reject(new Error('Browser deadline exceeded'));chrome.kill('SIGKILL');server.close();socket?.close();},60000);
try{
 const endpoint=await new Promise((resolve,reject)=>{let log='';const timer=setTimeout(()=>reject(new Error('Chrome startup timeout')),15000);chrome.on('error',reject);chrome.on('exit',code=>reject(new Error('Chrome exited '+code)));chrome.stderr.on('data',chunk=>{log+=chunk;const match=log.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(match){clearTimeout(timer);resolve(match[1]);}});});
 socket=new WebSocket(endpoint);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
 socket.onmessage=({data})=>{const value=JSON.parse(data);if(value.id){const item=pending.get(value.id);if(item){pending.delete(value.id);value.error?item.reject(new Error(value.error.message)):item.resolve(value.result);}}else if(value.method==='Runtime.exceptionThrown')errors.push(value.params.exceptionDetails.text);};
 function send(method,params={},sessionId=session){const id=++sequence;return new Promise((resolve,reject)=>{pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}: {})}));});}
 const version=await send('Browser.getVersion',{},undefined);const target=await send('Target.createTarget',{url:'about:blank'},undefined);session=(await send('Target.attachToTarget',{targetId:target.targetId,flatten:true},undefined)).sessionId;
 await send('Page.enable');await send('Runtime.enable');
 async function evaluate(expression){const result=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw new Error('Browser evaluation failed');return result.result.value;}
 async function wait(expression){const deadline=Date.now()+10000;while(Date.now()<deadline){if(await evaluate(expression))return;await new Promise(resolve=>setTimeout(resolve,100));}throw new Error('Browser condition timeout: '+expression);}
 async function assert(expression,label){if(!await evaluate(expression))throw new Error(label);}
 async function click(selector){await evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);}
 async function screenshot(name){await wait(`Array.from(document.querySelectorAll('.hero-render img,.current-preview img,.comparison img')).every(i=>i.complete&&i.naturalWidth>0)`);const result=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(path.join(evidence,name),Buffer.from(result.data,'base64'));}
 await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});await send('Page.navigate',{url:'http://127.0.0.1:'+server.address().port+(current?'/historical.html':'/')});
 await wait(`document.querySelectorAll('.family-card').length===12 && document.querySelector('#load-error').hidden===true`);
 await assert(`document.querySelectorAll('#view-metrics .badge.failed').length===1`,'Vase failure must remain visible');
 await assert(`document.querySelector('[data-mode="neutral"]').getAttribute('aria-pressed')==='true' && document.querySelector('#output-image img').getAttribute('src')==='assets/smooth_vase-inspection-oblique_35_28.png'`,'Actual shaded vase must be the default');
 await assert(`document.querySelector('#reason-body').textContent.includes('0.789001') && document.querySelector('#reason-body').textContent.includes('0.800000') && document.querySelector('#reason-facts').textContent.includes('998')`,'Exact vase failure must be explained');
 await screenshot('desktop.png');
 await click('#inspect-failure');await wait(`document.querySelector('#difference-canvas')?.dataset.missingPixels==='998'`);
 await assert(`document.querySelector('#difference-counts').textContent.includes('7,819') && document.querySelector('#difference-counts').textContent.includes('9,910')`,'Boundary bands must reproduce the scored overlap');
 await assert(`(()=>{const canvas=document.querySelector('#difference-canvas'),p=JSON.parse(canvas.dataset.checkPixel),color=canvas.getContext('2d').getImageData(...p,1,1).data;return color[0]===203&&color[1]===72&&color[2]===112})()`,'Exact reference-only pixel must be painted pink');
 await evaluate(`document.querySelector('.comparison').scrollIntoView({behavior:'instant'})`);await screenshot('vase-difference.png');
 await click('[data-layer="filled"]');await wait(`document.querySelector('#difference-canvas')?.dataset.layer==='filled'`);await assert(`document.querySelector('#difference-counts').textContent.includes('998 missing') && document.querySelector('#difference-counts').textContent.includes('0 extra')`,'Filled difference counts must be exact');
 await click('[data-case="rounded_triangle_dot"]');await wait(`document.querySelector('#output-image img')?.getAttribute('src')==='assets/rounded_triangle_dot-inspection-oblique_35_28.png'`);
 await assert(`document.querySelectorAll('#view-metrics .badge.passed').length===5 && document.querySelector('#surface-note').textContent.includes('unqualified') && document.querySelector('#reason-title').textContent.includes('unfinished') && document.querySelector('#reason-body').textContent.includes('no pass/fail threshold')`,'Triangle missing qualification must be distinct from measured failure');
 await evaluate(`document.querySelector('#inspection').scrollIntoView({behavior:'instant'})`);await screenshot('triangle-rendered.png');
 await click('[data-mode="mask"]');await assert(`document.querySelector('#output-image img').getAttribute('src')==='assets/rounded_triangle_dot-output-oblique_35_28.png'`,'Silhouette diagnostics must remain accessible');
 await click('[data-mode="neutral"]');await click('[data-view="top"]');await assert(`document.querySelector('#output-image .unavailable')!==null`,'An unretained camera pass must remain explicit');
 await click('[data-filter="reference"]');await assert(`document.querySelectorAll('.family-card').length===10`,'Reference-only filter incorrect');
 await click('[data-family="sphere"]');await assert(`document.querySelector('#output-image .unavailable')!==null && document.querySelectorAll('#view-metrics .badge.unmeasured').length===5 && document.querySelector('#reason-facts').textContent.includes('Not run')`,'Unrun reference cannot appear as failed output');
 await click('[data-filter="actual"]');await assert(`document.querySelectorAll('.family-card').length===2`,'Actual reconstruction filter incorrect');
 await click('[data-filter="all"]');await evaluate("document.querySelector('#families').scrollIntoView({behavior:'instant'})");await wait("Array.from(document.querySelectorAll('.family-card img')).every(image=>image.complete&&image.naturalWidth>0)");await screenshot('family-library.png');
 await assert(`document.querySelector('#load-error').hidden===true`,'Results failed to load');if(errors.length)throw new Error('Browser runtime exceptions: '+errors.join('; '));
 if(current){
  await send('Page.navigate',{url:'http://127.0.0.1:'+server.address().port+'/'});
  await wait(`document.body.dataset.state==='ready' && document.querySelectorAll('#family-grid [data-family]').length===12`);
  const engineering=current.families.filter(f=>['silhouette','boundary','topology','editability','surfaceEngineering'].every(g=>f.verdicts[g]==='passed')).length;
  const accepted=current.families.filter(f=>f.verdicts.aggregate==='passed').length;
  const gaps=current.families.filter(f=>f.verdicts.artistSurface==='unqualified').length;
  await assert(`document.querySelector('#family-total').textContent.trim()==='12' && document.querySelector('#engineering-total').textContent.trim()===${JSON.stringify(engineering.toString())} && document.querySelector('#accepted-total').textContent.trim()===${JSON.stringify(accepted.toString())} && document.querySelector('#artist-gaps').textContent.trim()===${JSON.stringify(gaps.toString())} && document.querySelector('#views-total').textContent.trim()===${JSON.stringify(current.families.reduce((n,f)=>n+f.viewMetrics.filter(m=>m.passed).length,0).toString())}`,'Current counts must match independent retained results');
  await assert(`document.querySelector('#inspection').dataset.family==='smooth_vase' && document.querySelector('#selected-geometry').textContent.includes(${JSON.stringify(current.families.find(f=>f.id==='smooth_vase').geometryHash)})`,'Main demo must default to the accepted calibrated vase');
  await wait(`document.querySelector('#hero-image img')?.complete===true`);
  await assert(`(()=>{const image=document.querySelector('#hero-image img'),slot=document.querySelector('#hero-image');if(!image||!image.naturalWidth)return false;const i=image.getBoundingClientRect(),s=slot.getBoundingClientRect();return getComputedStyle(image).objectFit==='contain'&&i.width<=s.width+1&&i.height<=s.height+1&&i.top>=s.top-1&&i.bottom<=s.bottom+1})()`,'Current hero must retain the recorded image without additional CSS cropping');
  await screenshot('current-main.png');
  for(const family of current.families){
   await evaluate(`(()=>{const select=document.querySelector('#family-select');select.value=${JSON.stringify(family.id)};select.dispatchEvent(new Event('change',{bubbles:true}));})()`);
   await wait(`document.querySelector('#inspection').dataset.family===${JSON.stringify(family.id)}`);
   await assert(`document.querySelector('#inspection').dataset.geometryHash===${JSON.stringify(family.geometryHash)} && document.querySelector('#selected-geometry').textContent.includes(${JSON.stringify(family.geometryHash)})`,'Current selector body mismatch: '+family.id);
   for(const [gate,status] of Object.entries(family.verdicts))await assert(`document.querySelector('#verdicts [data-verdict="'+${JSON.stringify(gate)}+'"]').dataset.status===${JSON.stringify(status)}`,'Independent current verdict mismatch: '+family.id+'/'+gate);
   await assert(`document.querySelectorAll('#view-metrics tr[data-view]').length===5`,'Missing current original-view metrics: '+family.id);
   for(const metric of family.viewMetrics)await assert(`document.querySelector('#view-metrics tr[data-view="'+${JSON.stringify(metric.view)}+'"]').dataset.status===${JSON.stringify(metric.passed?'passed':'failed')}`,'Original-view verdict mismatch: '+family.id+'/'+metric.view);
   for(const role of ['reference','candidate']){
    const expected=role==='reference'?family.sourceGeometryHash:family.geometryHash;
    await assert(`Array.from(document.querySelectorAll('#'+${JSON.stringify(role==='reference'?'reference-image':'candidate-image')}+' [data-role]')).every(i=>i.dataset.geometryHash===${JSON.stringify(expected)})`,'Current rendered image body mismatch: '+family.id+'/'+role);
   }
  }
  const paired=f=>f.images.some(i=>i.role==='reference'&&f.images.some(j=>j.role==='candidate'&&j.view===i.view&&j.style===i.style));
  const pictured=current.families.find(f=>paired(f)&&f.images.some(i=>i.style==='normals'))||current.families.find(paired);
  if(pictured){
   await click('#family-grid [data-family="'+pictured.id+'"]');
   await wait(`document.querySelector('#inspection').dataset.family===${JSON.stringify(pictured.id)}`);
   const pairs=pictured.images.filter(i=>i.role==='reference'&&pictured.images.some(j=>j.role==='candidate'&&j.view===i.view&&j.style===i.style));
   const pass=pairs.find(i=>i.style==='normals')||pairs[0];
   await evaluate(`(()=>{const view=document.querySelector('#view-select');view.value=${JSON.stringify(pass.view)};view.dispatchEvent(new Event('change',{bubbles:true}));const style=document.querySelector('#style-select');style.value=${JSON.stringify(pass.style)};style.dispatchEvent(new Event('change',{bubbles:true}));})()`);
   for(const role of ['reference','candidate']){
    const record=pictured.images.find(i=>i.role===role&&i.view===pass.view&&i.style===pass.style);
    await assert(`(()=>{const image=document.querySelector('#'+${JSON.stringify(role==='reference'?'reference-image':'candidate-image')}+' img');return image&&image.getAttribute('src')===${JSON.stringify(record.path)}&&image.dataset.role===${JSON.stringify(role)}&&image.dataset.view===${JSON.stringify(pass.view)}&&image.dataset.style===${JSON.stringify(pass.style)}&&image.dataset.geometryHash===${JSON.stringify(record.geometryHash)}})()`,'Current camera/pass control must display the exact retained pair: '+role);
   }
   await evaluate(`document.querySelector('#inspection').scrollIntoView({behavior:'instant'})`);
   await screenshot('current-paired-inspection.png');
   const missing=current.families[0].viewMetrics.map(v=>v.view).flatMap(view=>['neutral','normals','mask'].map(style=>({view,style}))).find(slot=>!pictured.images.some(i=>i.view===slot.view&&i.style===slot.style));
   if(missing){
    await evaluate(`(()=>{const view=document.querySelector('#view-select');view.value=${JSON.stringify(missing.view)};view.dispatchEvent(new Event('change',{bubbles:true}));const style=document.querySelector('#style-select');style.value=${JSON.stringify(missing.style)};style.dispatchEvent(new Event('change',{bubbles:true}));})()`);
    await assert(`document.querySelector('#reference-image .unavailable')&&document.querySelector('#candidate-image .unavailable')&&document.querySelector('#pair-status').textContent.includes('No matched comparison')`,'An unretained current camera/pass must stay explicit');
   }
  }
  const unavailable=current.families.find(f=>!f.images.length);
  if(unavailable){await click('#family-grid [data-family="'+unavailable.id+'"]');await assert(`!document.querySelector('#candidate-image img') && document.querySelector('#candidate-image').textContent.trim().length>0`,'Unavailable current pair must not substitute an old body');}
  if(errors.length)throw new Error('Current browser runtime exceptions: '+errors.join('; '));
 }
 await fs.writeFile(path.join(evidence,'result.json'),JSON.stringify({passed:true,chrome:version.product,checks:['desktop shaded actual defaults','exact measured vase failure explanation','scored boundary-band overlay and magnified pixels','exact filled-mask differences','triangle shaded output and unrun qualification distinction','secondary silhouettes and unavailable camera pass','unrun reference and actual filters','no browser exceptions',...(current?['main defaults to current calibrated vase','twelve exact current selector bodies and independent gates','current count/metric consistency','current image identity and unavailable-pair handling']:[])]},null,2)+'\n');
 console.log('Browser rendering and interaction checks passed on '+version.product);
}finally{clearTimeout(timer);if(socket)socket.close();chrome.kill();server.close();}
