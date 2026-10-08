#!/usr/bin/env node
/** Bounded hosted-runner browser check; uses existing Chrome, no install. */
import http from 'node:http';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
const directory=path.resolve('temp/pages-dist'), evidence=path.resolve('temp/pages-browser');
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
 async function screenshot(name){await wait(`Array.from(document.querySelectorAll('.hero-render img,.comparison img')).every(i=>i.complete&&i.naturalWidth>0)`);const result=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(path.join(evidence,name),Buffer.from(result.data,'base64'));}
 await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});await send('Page.navigate',{url:'http://127.0.0.1:'+server.address().port+'/'});
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
 await fs.writeFile(path.join(evidence,'result.json'),JSON.stringify({passed:true,chrome:version.product,checks:['desktop shaded actual defaults','exact measured vase failure explanation','scored boundary-band overlay and magnified pixels','exact filled-mask differences','triangle shaded output and unrun qualification distinction','secondary silhouettes and unavailable camera pass','unrun reference and actual filters','no browser exceptions']},null,2)+'\n');
 console.log('Browser rendering and interaction checks passed on '+version.product);
}finally{clearTimeout(timer);if(socket)socket.close();chrome.kill();server.close();}
