const $ = selector => document.querySelector(selector);
const escape = value => String(value).replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const labels = {front:'Front',side:'Side',top:'Top',oblique_35_28:'Oblique 35° / 28°',oblique_145_40:'Oblique 145° / 40°'};
const statusLabel = {passed:'Pass',failed:'Fail',unqualified:'Not qualified',unmeasured:'Not run',reference:'Reference only'};
const badge = (status, label=statusLabel[status]) => `<span class="badge ${escape(status)}">${escape(label)}</span>`;
let data, diagnostics, current='smooth_vase', view='oblique_35_28', mode='neutral', filter='all', layer='boundary', generation=0;

function image(slot, path, alt, unavailable='This pass was not retained. Choose another camera or inspect the silhouettes.') {
  const container=$(slot); container.replaceChildren();
  if (path) {const img=document.createElement('img');img.src=path;img.alt=alt;img.width=512;img.height=512;container.append(img);}
  else container.innerHTML=`<div class="unavailable"><span class="eyebrow">EVIDENCE UNAVAILABLE</span><strong>Not retained</strong><p>${escape(unavailable)}</p></div>`;
}
function select(id) {
  current=id; const family=data.families.find(item=>item.id===id);
  mode=family.output?'neutral':'mask'; view=family.output?'oblique_35_28':'front';render();
}
function diagnosis(family) {
  const limits=data.silhouetteLimits;
  if(family.status==='failed') {
    const failedView=data.views.find(camera=>!family.metrics[camera].passed), failed=family.metrics[failedView];
    $('#reason-title').textContent='A small outline mismatch crosses one cutoff.';
    $('#reason-body').innerHTML=`<p>The filled shape overlaps by <strong>${(failed.area_iou*100).toFixed(2)}%</strong> in ${escape(labels[failedView])}. The outline test scores <strong>${failed.boundary_iou.toFixed(6)}</strong> and needs <strong>at least ${limits.boundary.toFixed(6)}</strong> — a ${((limits.boundary-failed.boundary_iou)*100).toFixed(2)} percentage-point shortfall.</p><p>The outline score compares narrow bands around the edges. A thin inset can change those bands noticeably while barely changing the filled shape. The surface, actual solid-boundary check and artist-edit checks pass. This result is flagged by that one required outline test.</p>`;
    const row=diagnostics.cases[family.id][failedView];
    $('#reason-facts').textContent=`${row.missingPixels.toLocaleString()} reference foreground pixels missing; ${row.extraPixels.toLocaleString()} extra output pixels. ${row.bandIntersection.toLocaleString()} / ${row.bandUnion.toLocaleString()} scored boundary-band pixels overlap. No alignment or threshold change.`;
    $('#inspect-failure').hidden=false;$('#inspect-failure').dataset.failureView=failedView;
    $('#reason-status').innerHTML=badge('failed','Measured failure');
  } else if(family.output) {
    $('#reason-title').textContent='The outline passes. Surface qualification is unfinished.';
    $('#reason-body').innerHTML=`<p>All five silhouettes and the actual solid-boundary check pass. Surface measurements are <strong>${family.surface.symmetric_mean_distance_world.toFixed(8)} world units</strong> mean distance and <strong>${family.surface.normal_angle_p95_degrees.toFixed(5)}°</strong> normal P95.</p><p>We have not established this shape’s reference-mesh error floor or independent surface acceptance limits. Those measurements therefore have no pass/fail threshold yet. This is a qualification that has not been completed.</p>`;
    $('#reason-facts').textContent='The vase’s 0.003 / 2.5° limits apply to the vase only. No triangle surface failure or pass is inferred from them.';
    $('#inspect-failure').hidden=true;$('#reason-status').innerHTML=badge('unqualified','Qualification not completed');
  } else {
    $('#reason-title').textContent='This family has not been reconstructed.';
    $('#reason-body').innerHTML='<p>The pictured shape is an authored synthetic reference. There is no actual reconstructed output to compare, so no silhouette scores or reconstructed surface/solid/edit verdicts have been produced.</p><p>Reference preparation checks do not establish reconstruction acceptance.</p>';
    $('#reason-facts').textContent='Not run. No metric value or cutoff comparison exists for an actual output; this is not a measured failure.';
    $('#inspect-failure').hidden=true;$('#reason-status').innerHTML=badge('unmeasured','Reconstruction not run');
  }
  $('#qualification-status').textContent=family.qualification?`Actual solid-boundary check: completed and qualified (${family.qualification.elapsedSeconds.toFixed(2)} s, including identity and transfer). It did not time out.`:'Actual reconstruction checks: not run.';
}
function render() {
  const family=data.families.find(item=>item.id===current); if(!family)throw new Error('Unknown family');
  const token=++generation;
  $('#case-title').textContent=family.label;$('#case-summary').textContent=family.summary;$('#camera-label').textContent=labels[view];
  $('#case-status').className=`badge ${family.status}`;$('#case-status').textContent=family.statusLabel;
  const isRender=mode==='neutral';
  const reference=isRender?family.referenceInspection?.[view]:family.reference?.[view];
  const output=isRender?family.inspection?.[view]:family.output?.[view];
  image('#reference-image',reference,`${family.label}: authored reference ${labels[view]}, ${isRender?'shaded object':'silhouette'}`);
  image('#output-image',output,`${family.label}: actual reconstructed ${labels[view]}, ${isRender?'shaded object':'silhouette'}`,family.output?'No shaded pass was retained for this camera. Silhouette evidence is available.':'Reconstruction not run. An authored reference is not an actual output.');
  $('#reference-caption').textContent=isRender?'Authored reference. Shading is for inspection; geometry metrics use retained coordinates.':'Frozen reference mask from the original shared camera.';
  $('#output-caption').textContent=output?(isRender?'Actual reconstructed geometry, shaded for inspection. No geometry or acceptance metric changed.':'Actual output mask compared with the frozen source.'):'No image was substituted for unavailable output evidence.';
  $('#frame-note').textContent=view.startsWith('oblique')?(family.borderNote||'Frozen oblique reference; actual reconstruction remains unavailable.'):'Front, side and top are source views; both obliques are retained inspection / held-out views.';
  $('#view-buttons').innerHTML=data.views.map(camera=>`<button data-view="${camera}" aria-pressed="${camera===view}">${escape(labels[camera])}</button>`).join('');
  document.querySelectorAll('[data-case]').forEach(button=>button.setAttribute('aria-pressed',button.dataset.case===current));
  document.querySelectorAll('[data-mode]').forEach(button=>button.setAttribute('aria-pressed',button.dataset.mode===mode));
  $('#verdict-strip').innerHTML=Object.entries(family.verdicts).map(([key,status])=>`<div><span>${escape({silhouette:'Outline',surface:'Surface',boundary:'Solid geometry',editability:'Artist edits'}[key])}</span>${badge(status)}</div>`).join('');
  $('#view-metrics').innerHTML=data.views.map(camera=>{const metric=family.metrics?.[camera];return `<tr class="${camera===view?'active-row':''}"><td>${escape(labels[camera])}</td>${['area_iou','boundary_iou','signed_distance_loss'].map((key,index)=>`<td class="${metric&&(index===0?metric[key]<data.silhouetteLimits.area:index===1?metric[key]<data.silhouetteLimits.boundary:metric[key]>data.silhouetteLimits.distance)?'metric-failed':''}">${metric?metric[key].toFixed(6):'—'}</td>`).join('')}<td>${badge(metric?(metric.passed?'passed':'failed'):'unmeasured')}</td></tr>`}).join('');
  $('#surface-values').innerHTML=family.surface?`<div class="surface-value">${family.surface.symmetric_mean_distance_world.toFixed(8)}<span>Mean surface distance / world units</span></div><div class="surface-value">${family.surface.normal_angle_p95_degrees.toFixed(5)}°<span>Geometric normal angle / P95</span></div>`:'<p class="quiet">Surface reconstruction measurements have not been run.</p>';
  $('#surface-note').textContent=family.surface?(family.surface.limits?'Vase only: mean ≤ 0.003; normal P95 ≤ 2.5°. Both pass.':'Triangle reference error and family limits remain unqualified. No surface threshold or pass/fail verdict exists yet.'):'Actual reconstruction and independent family surface qualification remain unrun.';
  $('#edit-scope').textContent=family.editScope||'The reference has an index/volume screen and a live object-scale response. Actual reconstructed semantic edits have not been run.';
  $('#geometry-identity').innerHTML=`<dt>Reference geometry hash</dt><dd>${escape(family.referenceGeometry)}</dd>${family.geometry?`<dt>Actual geometry hash</dt><dd>${escape(family.geometry)}</dd><dt>Actual OBJ SHA-256</dt><dd>${escape(family.objSha256)}</dd>`:''}`;
  $('#case-evidence').href=current==='smooth_vase'?data.evidenceLinks.vase:current==='rounded_triangle_dot'?data.evidenceLinks.triangle:data.evidenceLinks.workload;
  diagnosis(family);
  $('#difference-tools').hidden=mode!=='difference';
  if(mode==='difference')paintDifference(family,token).catch(error=>{if(token===generation){$('#difference-counts').textContent='Difference evidence unavailable. Use the exact source receipt.';}console.error(error);});
  renderFamilies();
}
async function paintDifference(family,token) {
  const record=diagnostics.cases[family.id]?.[view];
  if(!record){$('#difference-counts').textContent='No actual output exists; difference evidence has not been run.';return;}
  const source=new Image();source.src=family.reference[view];await source.decode();if(token!==generation)return;
  const canvas=document.createElement('canvas');canvas.id='difference-canvas';canvas.width=record.width;canvas.height=record.height;
  canvas.setAttribute('aria-label',`Exact ${layer==='boundary'?'boundary-band':'filled-mask'} differences for ${family.label}, ${labels[view]}`);
  const context=canvas.getContext('2d');context.drawImage(source,0,0);
  const missing=layer==='boundary'?record.missingBandRuns:record.missingRuns;
  const extra=layer==='boundary'?record.extraBandRuns:record.extraRuns;
  context.fillStyle='#cb4870';for(const [y,x0,x1] of missing)context.fillRect(x0,y,x1-x0,1);
  context.fillStyle='#167eaa';for(const [y,x0,x1] of extra)context.fillRect(x0,y,x1-x0,1);
  canvas.dataset.missingPixels=record.missingPixels;canvas.dataset.extraPixels=record.extraPixels;canvas.dataset.layer=layer;
  if(missing.length)canvas.dataset.checkPixel=JSON.stringify([missing[0][1],missing[0][0]]);
  $('#output-image').replaceChildren(canvas);
  $('#output-caption').textContent=layer==='boundary'?'Pink: reference band only. Blue: output band only. Scored bands use the original radius-2 elliptical kernel.':'Pink: reference foreground missing in output. Blue: extra output foreground. Exact original pixel grid; no alignment.';
  $('#difference-counts').textContent=layer==='boundary'?`Boundary IoU ${record.boundaryIoU.toFixed(6)} = ${record.bandIntersection.toLocaleString()} overlapping / ${record.bandUnion.toLocaleString()} union band pixels. Cutoff ≥ ${data.silhouetteLimits.boundary.toFixed(6)}.`:`${record.missingPixels.toLocaleString()} missing foreground pixels; ${record.extraPixels.toLocaleString()} extra pixels. Filled-shape IoU ${record.areaIoU.toFixed(6)}.`;
  document.querySelectorAll('[data-layer]').forEach(button=>button.setAttribute('aria-pressed',button.dataset.layer===layer));
  function crop(x,y){const width=64,x0=Math.max(0,Math.min(record.width-width,Math.round(x)-width/2)),y0=Math.max(0,Math.min(record.height-width,Math.round(y)-width/2));const zoom=$('#difference-zoom').getContext('2d');zoom.imageSmoothingEnabled=false;zoom.clearRect(0,0,256,256);zoom.drawImage(canvas,x0,y0,width,width,0,0,256,256);$('#zoom-caption').textContent=`4× pixel crop · x ${x0}–${x0+width-1}, y ${y0}–${y0+width-1}. Move over the overlay to inspect another region.`;}
  crop(...record.focus);canvas.addEventListener('pointermove',event=>{const rect=canvas.getBoundingClientRect();crop((event.clientX-rect.left)/rect.width*record.width,(event.clientY-rect.top)/rect.height*record.height);});
}
function renderFamilies() {
  const shown=data.families.filter(family=>filter==='all'||(filter==='actual'?family.output:!family.output));
  $('#family-grid').innerHTML=shown.map(family=>`<button class="family-card" data-family="${family.id}" aria-pressed="${family.id===current}"><img src="${family.thumbnail}" alt="${escape(family.label)} ${family.output?'actual shaded reconstruction':'authored reference silhouette'}" width="512" height="512" loading="lazy"><div class="family-meta"><div class="family-top"><span>${String(data.families.indexOf(family)+1).padStart(2,'0')} / ${family.heldOut?'HELD-OUT':'SYNTHETIC'}</span>${badge(family.status,family.status==='reference'?'Not reconstructed':family.status==='failed'?'1 outline gate fails':'Qualification pending')}</div><h3>${escape(family.label)}</h3><p>${family.output?escape(family.summary):'Reference prepared. Reconstruction has not been run.'}</p></div><span class="inspect">Inspect evidence ↗</span></button>`).join('');
  $('#filter-count').textContent=`Showing ${shown.length} of 12 families. The ten reference-only families have not been reconstructed.`;
}
document.addEventListener('click',event=>{
  const button=event.target.closest('button');if(!button)return;
  if(button.id==='inspect-failure'){view=button.dataset.failureView;mode='difference';layer='boundary';render();$('#difference-tools').scrollIntoView({behavior:'instant',block:'center'});}
  if(button.dataset.case)select(button.dataset.case);
  if(button.dataset.view){view=button.dataset.view;render();}
  if(button.dataset.mode){mode=button.dataset.mode;render();}
  if(button.dataset.layer){layer=button.dataset.layer;render();}
  if(button.dataset.filter){filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(item=>item.setAttribute('aria-pressed',item.dataset.filter===filter));renderFamilies();}
  if(button.dataset.family){select(button.dataset.family);$('#inspection').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});}
});
try {
  const responses=await Promise.all([fetch('data.json'),fetch('diagnostics.json')]);if(responses.some(response=>!response.ok))throw new Error('Results unavailable');
  [data,diagnostics]=await Promise.all(responses.map(response=>response.json()));render();
  const depth=data.negativeControls.find(control=>control.id==='incorrect_depth');$('#depth-control').textContent=`Side area IoU ${depth.views.side.toFixed(6)} — incorrect thickness detected.`;
  const response=await fetch('build.json');if(response.ok){const build=await response.json();$('#build-id').textContent=`Measured evidence: ${data.evidenceDate} · source ${data.sourceSnapshot.slice(0,7)} · site build ${build.commit.slice(0,7)}`;}
} catch(error){$('#load-error').hidden=false;$('#load-error').textContent='Interactive evidence could not load. Use the repository receipts below.';console.error(error);}
