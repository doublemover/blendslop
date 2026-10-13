import fs from 'node:fs';import path from 'node:path';import vm from 'node:vm';import assert from 'node:assert/strict';
import {JSDOM} from 'jsdom';import {createCanvas,Image} from '@napi-rs/canvas';
const root=path.resolve(process.argv[2]);
const dom=new JSDOM(fs.readFileSync(path.join(root,'historical.html'),'utf8'),{url:'http://localhost/historical.html',runScripts:'outside-only',pretendToBeVisual:true});
const w=dom.window, errors=[];
w.console.error=(...args)=>errors.push(args.map(String).join(' '));w.matchMedia=()=>({matches:false});w.HTMLElement.prototype.scrollIntoView=function(){};
w.fetch=async url=>{const p=path.join(root,url);return{ok:fs.existsSync(p),json:async()=>JSON.parse(fs.readFileSync(p,'utf8'))};};
const backing=new WeakMap();
w.HTMLCanvasElement.prototype.getContext=function(kind){assert.equal(kind,'2d');let c=backing.get(this);if(!c||c.width!==this.width||c.height!==this.height){c=createCanvas(this.width,this.height);backing.set(this,c);}const ctx=c.getContext('2d');if(!ctx.__wrapped){const draw=ctx.drawImage.bind(ctx);ctx.drawImage=(image,...args)=>draw(backing.get(image)||image,...args);ctx.__wrapped=true;}return ctx;};
w.HTMLCanvasElement.prototype.getBoundingClientRect=function(){return {left:0,top:0,width:this.width,height:this.height,right:this.width,bottom:this.height};};
let decodeCount=0,deferDecode=false;const pendingDecodes=[];
w.Image=class extends Image{set src(value){super.src=fs.readFileSync(path.join(root,value));}decode(){decodeCount++;return deferDecode?new Promise((resolve,reject)=>pendingDecodes.push({resolve,reject})):Promise.resolve();}};
const context=dom.getInternalVMContext(),cache=new Map();async function load(file){if(cache.has(file))return cache.get(file);const mod=new vm.SourceTextModule(fs.readFileSync(file,'utf8'),{context,identifier:file});cache.set(file,mod);await mod.link(spec=>load(path.resolve(path.dirname(file),spec)));return mod;}
await (await load(path.join(root,'app.js'))).evaluate();
const $=s=>w.document.querySelector(s);async function click(s){assert.ok($(s),s);$(s).dispatchEvent(new w.MouseEvent('click',{bubbles:true}));await new Promise(r=>setTimeout(r,0));}
await click('#inspect-failure');assert.equal($('#difference-canvas').dataset.missingPixels,'998');await click('[data-layer="outline"]');
assert.equal($('#reference-label').textContent,'Detail');assert.equal($('#output-label').textContent,'Comparison');assert.equal($('#difference-canvas').tabIndex,0);
const stableCanvas=$('#difference-canvas'),decodesBeforeZoom=decodeCount;
for(const zoom of [2,4,8]){await click(`[data-zoom="${zoom}"]`);assert.equal(JSON.parse($('#difference-zoom').dataset.crop)[2],512/zoom);assert.equal($(`[data-zoom="${zoom}"]`).getAttribute('aria-pressed'),'true');}
assert.equal($('#difference-canvas'),stableCanvas);assert.equal(decodeCount,decodesBeforeZoom,'Zoom reuses decoded source and contour geometry');
$('#difference-canvas').dispatchEvent(new w.MouseEvent('pointerdown',{clientX:256,clientY:256,bubbles:true}));$('#difference-canvas').dispatchEvent(new w.KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));assert.equal(JSON.parse($('#difference-zoom').dataset.crop)[0],228);
const before=JSON.parse($('#difference-zoom').dataset.crop);await click('[data-layer="boundary"]');assert.deepEqual(JSON.parse($('#difference-zoom').dataset.crop),before);
const pixel=JSON.parse($('#difference-canvas').dataset.checkPixel);assert.deepEqual(Array.from($('#difference-canvas').getContext('2d').getImageData(...pixel,1,1).data),[203,72,112,255]);
await click('[data-mode="neutral"]');assert.equal($('.comparison').classList.contains('is-comparing'),false);assert.equal($('#zoom-controls').hidden,true);
await click('[data-case="rounded_triangle_dot"]');await click('[data-mode="difference"]');assert.ok($('#difference-canvas'));assert.ok($('#difference-zoom'));assert.equal($('#case-title').textContent,'Rounded triangular dot');
await click('[data-family="sphere"]');assert.equal($('#zoom-controls').hidden,true);assert.equal($('#difference-canvas'),null);assert.ok($('#output-image .unavailable'));
await click('[data-case="smooth_vase"]');await click('#inspect-failure');await click('[data-layer="filled"]');assert.ok($('#difference-counts').textContent.includes('998 missing'));assert.equal(errors.length,0,errors.join('\n'));
const held=JSON.parse($('#difference-zoom').dataset.crop);$('#difference-canvas').dispatchEvent(new w.KeyboardEvent('keydown',{key:'ArrowRight',ctrlKey:true,bubbles:true}));assert.deepEqual(JSON.parse($('#difference-zoom').dataset.crop),held);
deferDecode=true;await click('[data-layer=\"outline\"]');assert.match($('#difference-counts').textContent,/Loading/);await click('[data-case=\"rounded_triangle_dot\"]');pendingDecodes.shift().resolve();await new Promise(r=>setTimeout(r,0));assert.equal($('#difference-canvas'),null);assert.equal($('#case-title').textContent,'Rounded triangular dot');
await click('[data-mode=\"difference\"]');await click('[data-case=\"smooth_vase\"]');pendingDecodes.shift().reject(new Error('stale decode fixture'));await new Promise(r=>setTimeout(r,0));assert.equal($('#difference-canvas'),null);assert.equal(errors.length,0,'Stale image failures cannot corrupt the current selection');
const out={passed:true,environment:'jsdom with real native Canvas2D rasterization, not browser layout',checks:['real retained PNGs load','998-pixel failure identity','outline and detail roles','2x/4x/8x exact crop sizes','pointer and keyboard crop movement','layer change preserves crop','exact boundary pixel RGBA','leaving comparison restores normal mode','family changes reset compatible controls','reference-only family has no fake output','repeated return to failure remains exact','zoom does not decode or rebuild canvases','stale decode and failure cannot overwrite a new family','modified arrow shortcuts preserved','no app errors']};console.log(JSON.stringify(out,null,2));dom.window.close();
