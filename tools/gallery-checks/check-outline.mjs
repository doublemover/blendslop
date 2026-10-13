import assert from 'node:assert/strict';
import {maskContours} from '../../site/outline.js';
let edges=0;
for(let bits=0;bits<512;bits++){
 const mask=Uint8Array.from({length:9},(_,i)=>(bits>>i)&1), paths=maskContours(mask,3,3);let twiceArea=0;
 for(const p of paths){assert.deepEqual(p[0],p.at(-1));for(let i=0;i<p.length-1;i++){const[a,b]=p[i],[c,d]=p[i+1];assert.equal(Math.abs(a-c)+Math.abs(b-d),1);twiceArea+=a*d-c*b;edges++;}}
 assert.equal(twiceArea/2,mask.reduce((a,b)=>a+b,0));
}
console.log(JSON.stringify({passed:true,exhaustive3x3Masks:512,orientedUnitEdges:edges,contracts:['closed contours','axis-aligned unit edges','exact foreground area including holes/diagonal contacts']}));
