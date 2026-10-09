const $ = selector => document.querySelector(selector);
const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const views = ['front','side','top','oblique_35_28','oblique_145_40'];
const viewLabels = {front:'Front',side:'Side',top:'Top',oblique_35_28:'Oblique 35 / 28',oblique_145_40:'Oblique 145 / 40'};
const styleLabels = {neutral:'Neutral shading',normals:'Shading normals',mask:'Silhouette'};
const verdictLabels = {silhouette:'Outline / five views',boundary:'Exact solid boundary',topology:'Index topology',editability:'Semantic edit / restore',surfaceEngineering:'Surface engineering',artistSurface:'Artist surface criteria',aggregate:'Aggregate acceptance'};
const engineeringKeys = ['silhouette','boundary','topology','editability','surfaceEngineering'];
const verdictKeys = [...engineeringKeys,'artistSurface','aggregate'];
const statusLabels = {passed:'Passed',failed:'Failed',unqualified:'Unqualified'};
const hashPattern = /^[a-f0-9]{64}$/;
let data, selected, view='oblique_35_28', style='neutral', filter='all';

function engineeringComplete(family) {
    return engineeringKeys.every(key => family.verdicts[key] === 'passed');
}
function headline(family) {
    if (engineeringComplete(family)) return ['passed','Engineering complete'];
    if (engineeringKeys.some(key => family.verdicts[key] === 'failed')) return ['failed','Engineering failure retained'];
    return ['unqualified','Engineering gaps'];
}
function badge(status, label=statusLabels[status]) {
    return `<span class="badge ${escape(status)}">${escape(label)}</span>`;
}
function validImage(image, family) {
    return image && hashPattern.test(image.geometryHash) &&
        image.geometryHash === (image.role === 'reference' ? family.sourceGeometryHash : family.geometryHash);
}
function imageFor(family, role, selectedView=view, selectedStyle=style) {
    return family.images.find(image => image.role === role && image.view === selectedView &&
        image.style === selectedStyle && validImage(image,family));
}
function thumbnail(family) {
    return family.images.find(image => image.role === 'candidate' && image.style === 'neutral' && validImage(image,family)) ||
        family.images.find(image => image.role === 'candidate' && validImage(image,family));
}
function validate(snapshot) {
    if (snapshot.schema !== 1 || snapshot.protocol !== 'quality_latest_family_results_v1' ||
        snapshot.evidenceDate !== '2026-10-09' || !/^[a-f0-9]{40}$/.test(snapshot.sourceRevision) ||
        snapshot.scope?.historicalGallery !== 'historical.html' || snapshot.scope?.acceptedMeaning !== 'artist-defined aggregate acceptance' ||
        !Array.isArray(snapshot.families) || snapshot.families.length !== 12 || new Set(snapshot.families.map(row => row.id)).size !== 12) {
        throw new Error('Current snapshot identity or family inventory is invalid.');
    }
    for (const family of snapshot.families) {
        if (!/^[a-z_]+$/.test(family.id) || typeof family.label !== 'string' || !hashPattern.test(family.geometryHash) ||
            !hashPattern.test(family.sourceGeometryHash) || typeof family.checkpointLabel !== 'string' || !family.verdicts ||
            verdictKeys.some(key => !Object.hasOwn(statusLabels,family.verdicts[key])) ||
            !Array.isArray(family.viewMetrics) || !Array.isArray(family.images) || !Array.isArray(family.evidence) ||
            !Array.isArray(family.notes) || !family.surface || !Array.isArray(family.surface.notes)) {
            throw new Error('Current family record is invalid: '+family.id);
        }
        const metricViews = new Set();
        for (const metric of family.viewMetrics) {
            if (!views.includes(metric.view) || metricViews.has(metric.view) || typeof metric.passed !== 'boolean' ||
                !['area_iou','boundary_iou','signed_distance_loss'].every(key => Number.isFinite(metric[key]) && metric[key]>=0) ||
                metric.area_iou>1 || metric.boundary_iou>1 ||
                metric.passed !== (metric.area_iou>=.7 && metric.boundary_iou>=.8 && metric.signed_distance_loss<=.05)) {
                throw new Error('Per-view verdict differs from its unchanged gates: '+family.id);
            }
            metricViews.add(metric.view);
        }
        if (family.verdicts.silhouette === 'passed' && (metricViews.size !== 5 || family.viewMetrics.some(metric => !metric.passed))) {
            throw new Error('Passing outline lacks five passing views: '+family.id);
        }
        if (family.verdicts.aggregate === 'passed' && (family.verdicts.artistSurface !== 'passed' || !engineeringComplete(family))) {
            throw new Error('Aggregate pass lacks independent qualified checks: '+family.id);
        }
        for (const key of ['meanDistance','normalP95']) {
            if (family.surface[key] !== undefined && (!Number.isFinite(family.surface[key]) || family.surface[key]<0)) {
                throw new Error('Raw surface observation is invalid: '+family.id);
            }
        }
        if (![...family.notes,...family.surface.notes].every(note => typeof note === 'string')) throw new Error('Invalid scope notes.');
        const imageKeys = new Set();
        for (const image of family.images) {
            const key = image.role+'/'+image.view+'/'+image.style;
            if (!['reference','candidate'].includes(image.role) || !views.includes(image.view) ||
                typeof image.style !== 'string' || !/^[a-z_-]+$/.test(image.style) ||
                !/^assets\/[a-zA-Z0-9_/-]+\.png$/.test(image.path) || image.path.includes('..') ||
                !validImage(image,family) || imageKeys.has(key)) {
                throw new Error('Image is not bound to the declared selected body/pass: '+family.id);
            }
            imageKeys.add(key);
        }
        for (const item of family.evidence) {
            if (typeof item.label !== 'string' || typeof item.url !== 'string' || !/^https:\/\//.test(item.url)) {
                throw new Error('Invalid evidence link: '+family.id);
            }
        }
    }
}
function renderCounts() {
    $('#family-total').textContent = data.families.length;
    $('#engineering-total').textContent = data.families.filter(engineeringComplete).length;
    $('#views-total').textContent = data.families.reduce((sum,family) => sum+family.viewMetrics.filter(metric => metric.passed).length,0);
    $('#accepted-total').textContent = data.families.filter(family => family.verdicts.aggregate === 'passed').length;
    const artistPending = data.families.filter(family => family.verdicts.artistSurface === 'unqualified').length;
    $('#artist-gaps').textContent = artistPending;
    $('#snapshot-scope').textContent = `Evidence ${data.evidenceDate} / source ${data.sourceRevision.slice(0,7)} / ${artistPending} families await independent artist surface criteria. Negative controls are separate.`;
    $('#build-id').textContent = `Current selected evidence: ${data.sourceRevision}. Image and verdict bindings retain their original scope.`;
}
function renderFamilies() {
    const shown = data.families.filter(family => filter === 'all' || (filter === 'complete' ? engineeringComplete(family) : !engineeringComplete(family)));
    $('#family-grid').innerHTML = shown.length ? shown.map(family => {
        const image=thumbnail(family), [status,label]=headline(family);
        return `<button class="family-card" data-family="${family.id}" aria-pressed="${family.id===selected}" aria-label="Inspect ${escape(family.label)}"><div class="family-image">${image?`<img src="${escape(image.path)}" alt="${escape(family.label)}: selected reconstruction, ${escape(styleLabels[image.style]||image.style)}" width="512" height="512" loading="lazy">`:'<span class="no-preview">Current geometry retained.<br>Preview unavailable.</span>'}</div><div class="family-meta"><h3>${escape(family.label)}</h3>${badge(status,label)}<p>Artist aggregate: ${escape(statusLabels[family.verdicts.aggregate])}</p><span class="family-hash">Geometry ${family.geometryHash.slice(0,12)}</span></div></button>`;
    }).join('') : `<p class="empty-filter">${filter==='gaps'?'No engineering gaps remain in the selected rows.':'No selected row has all five engineering checks passed.'}</p>`;
    $('#filter-count').textContent = `Showing ${shown.length} of ${data.families.length} selected reconstructions. Engineering completion does not imply artist acceptance.`;
    document.querySelectorAll('[data-filter]').forEach(button => button.setAttribute('aria-pressed',button.dataset.filter===filter));
}
function renderList(selector, items, empty) {
    const list=$(selector);
    list.replaceChildren();
    for (const text of items.length?items:[empty]) {
        const item=document.createElement('li');item.textContent=text;list.append(item);
    }
}
function unavailable(slot, detail) {
    $(slot).innerHTML = `<div class="unavailable"><span class="eyebrow">EXACT PASS UNAVAILABLE</span><strong>No matched preview</strong><p>${escape(detail)}</p></div>`;
}
function renderImage(slot, image, alt) {
    const element=document.createElement('img');element.src=image.path;element.alt=alt;element.width=512;element.height=512;
    element.dataset.role=image.role;element.dataset.view=image.view;element.dataset.style=image.style;element.dataset.geometryHash=image.geometryHash;
    element.addEventListener('error',() => {
        if ($(slot).contains(element)) unavailable(slot,'The declared image could not be loaded. Its linked receipt remains the source of truth.');
    },{once:true});
    $(slot).replaceChildren(element);
}
function renderComparison(family) {
    const source=imageFor(family,'reference'), candidate=imageFor(family,'candidate');
    $('#reference-pass').textContent = $('#candidate-pass').textContent = `${viewLabels[view]} / ${styleLabels[style]||style}`;
    if (source) {
        renderImage('#reference-image',source,`${family.label}: authored reference, ${viewLabels[view]}, ${styleLabels[style]||style}`);
        $('#reference-caption').textContent = `Reference geometry ${source.geometryHash}`;
    } else {
        unavailable('#reference-image','The authored reference pass for this exact view, style and source geometry was not retained.');
        $('#reference-caption').textContent = 'Reference pass unavailable.';
    }
    if (candidate) {
        renderImage('#candidate-image',candidate,`${family.label}: exact selected reconstruction, ${viewLabels[view]}, ${styleLabels[style]||style}`);
        $('#candidate-caption').textContent = `Selected geometry ${candidate.geometryHash}`;
    } else {
        unavailable('#candidate-image','The candidate pass for this exact view, style and selected geometry was not retained.');
        $('#candidate-caption').textContent = 'Selected candidate pass unavailable.';
    }
    if (source && candidate) {
        $('#pair-status').textContent = 'Shared view and pass retained; each image is bound to its declared geometry.';
    } else {
        const missing = source?'candidate':candidate?'reference':'reference and candidate';
        $('#pair-status').textContent = `No matched comparison: ${missing} pass unavailable. Each retained role is shown independently; view metrics and verdicts remain visible.`;
    }
}
function renderSelection() {
    const family=data.families.find(row => row.id===selected), [status,label]=headline(family);
    $('#case-title').textContent = family.label;
    $('#checkpoint-label').textContent = family.checkpointLabel;
    $('#case-summary').textContent = `${label}. Artist surface criteria: ${statusLabels[family.verdicts.artistSurface].toLowerCase()}. Aggregate acceptance: ${statusLabels[family.verdicts.aggregate].toLowerCase()}.`;
    $('#selected-geometry').textContent = family.geometryHash;
    $('#inspection').dataset.family = family.id;
    $('#inspection').dataset.geometryHash = family.geometryHash;
    $('#hero-family-label').textContent = family.label+' / current geometry';
    $('#hero-acceptance').textContent = family.verdicts.aggregate==='passed'?'Artist aggregate accepted':label;
    const hero=imageFor(family,'candidate',view,'neutral')||thumbnail(family);
    if (hero) {
        renderImage('#hero-image',hero,`${family.label}: exact selected reconstruction, ${viewLabels[hero.view]}, ${styleLabels[hero.style]||hero.style}`);
        $('#hero-caption').textContent = `${viewLabels[hero.view]} / ${styleLabels[hero.style]||hero.style} / geometry ${family.geometryHash.slice(0,12)}. The recorded pass depicts this selected body.`;
    } else {
        unavailable('#hero-image','A preview for this exact selected geometry is unavailable. Its verdicts remain visible below.');
        $('#hero-caption').textContent = 'Selected geometry '+family.geometryHash;
    }
    $('#family-select').value = selected;
    $('#verdicts').innerHTML = verdictKeys.map(key => `<div class="verdict ${engineeringKeys.includes(key)?'engineering':'artist'}" data-verdict="${key}" data-status="${family.verdicts[key]}"><span>${escape(verdictLabels[key])}</span>${badge(family.verdicts[key])}</div>`).join('');
    $('#view-select').innerHTML = views.map(camera => `<option value="${camera}">${viewLabels[camera]}</option>`).join('');
    $('#view-select').value = view;
    const styles = [...new Set(['neutral','normals','mask',...family.images.map(image => image.style)])];
    if (!styles.includes(style)) style='neutral';
    $('#style-select').innerHTML = styles.map(pass => `<option value="${escape(pass)}">${escape(styleLabels[pass]||pass)}</option>`).join('');
    $('#style-select').value = style;
    $('#view-metrics').innerHTML = views.map(camera => {
        const metric=family.viewMetrics.find(row => row.view===camera);
        return `<tr data-view="${camera}" data-status="${metric?(metric.passed?'passed':'failed'):'unqualified'}" class="${camera===view?'active-row':''}"><th scope="row">${viewLabels[camera]}</th>${['area_iou','boundary_iou','signed_distance_loss'].map(key => `<td>${metric?metric[key].toFixed(6):'—'}</td>`).join('')}<td>${badge(metric?(metric.passed?'passed':'failed'):'unqualified',metric?(metric.passed?'Passed':'Failed'):'Not retained')}</td></tr>`;
    }).join('');
    $('#surface-values').innerHTML = [['meanDistance','Mean distance / world units'],['normalP95','Oriented normal angle / P95 degrees']].map(([key,title]) => `<div class="surface-observation"><strong>${family.surface[key]===undefined?'Not retained':family.surface[key].toFixed(key==='meanDistance'?9:5)}</strong><span>${title}</span></div>`).join('');
    renderList('#surface-notes',family.surface.notes,'No additional surface scope note was retained.');
    renderList('#case-notes',family.notes,'No additional case history was retained.');
    const evidence=$('#evidence-links');evidence.replaceChildren();
    for (const record of family.evidence) {
        const item=document.createElement('li'),link=document.createElement('a');link.textContent=record.label;link.href=record.url;item.append(link);evidence.append(item);
    }
    if (!family.evidence.length) renderList('#evidence-links',[],'Evidence link unavailable; no source receipt is substituted.');
    renderComparison(family);renderFamilies();
}
function select(familyId) {
    const family=data.families.find(row => row.id===familyId);
    if (!family) return;
    selected=familyId;
    view='oblique_35_28';style='neutral';
    if (!imageFor(family,'reference') || !imageFor(family,'candidate')) {
        const source=family.images.find(image => image.role==='reference' && imageFor(family,'candidate',image.view,image.style));
        if (source) {view=source.view;style=source.style;}
    }
    const url=new URL(location.href);url.searchParams.set('family',selected);history.replaceState(null,'',url);
    renderSelection();
}
document.addEventListener('click',event => {
    const button=event.target.closest('button');if (!button || !data) return;
    if (button.dataset.filter) {filter=button.dataset.filter;renderFamilies();}
    if (button.dataset.family) {select(button.dataset.family);$('#inspection').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});}
});
$('#family-select').addEventListener('change',event => select(event.target.value));
$('#view-select').addEventListener('change',event => {view=event.target.value;renderSelection();});
$('#style-select').addEventListener('change',event => {style=event.target.value;renderSelection();});
try {
    const response=await fetch('latest.json');
    if (!response.ok) throw new Error('Current results are not available in this build.');
    data=await response.json();validate(data);renderCounts();
    $('#family-select').innerHTML=data.families.map(family => `<option value="${family.id}">${escape(family.label)}</option>`).join('');
    $('#results-content').hidden=false;
    document.body.dataset.state='ready';
    const requested=new URL(location.href).searchParams.get('family');
    select(data.families.some(family => family.id===requested)?requested:data.families.some(family => family.id==='smooth_vase')?'smooth_vase':data.families[0].id);
} catch (error) {
    data=null;$('#results-content').hidden=true;
    document.body.dataset.state='error';
    $('#load-error').textContent=`Current result display unavailable: ${error.message} The historical gallery remains available. No result is inferred.`;
    $('#load-error').hidden=false;
    console.error(error);
}
