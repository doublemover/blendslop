/** Validate the current demo against retained evidence; no rendering or fitting. */
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const digest = /^[a-f0-9]{64}$/;
const views = ['front', 'side', 'top', 'oblique_35_28', 'oblique_145_40'];
const gates = ['silhouette', 'boundary', 'topology', 'editability', 'surfaceEngineering', 'artistSurface', 'aggregate'];
const states = new Set(['passed', 'failed', 'unqualified']);
function requireValue(condition, message) { if (!condition) throw new Error(message); }
export function pointerValue(value, pointer) {
    requireValue(typeof pointer === 'string' && pointer.startsWith('/'), 'Invalid evidence JSON pointer');
    for (const part of pointer.slice(1).split('/')) {
        const key = part.replace(/~1/g, '/').replace(/~0/g, '~');
        requireValue(value !== null && typeof value === 'object' && Object.hasOwn(value, key), 'Missing evidence pointer ' + pointer);
        value = value[key];
    }
    return value;
}
export async function validateLatest(root, source, historical) {
    let data;
    try { data = JSON.parse(await fs.readFile(path.join(source, 'latest.json'), 'utf8')); }
    catch (error) { if (error.code === 'ENOENT') return null; throw error; }
    const manifest = JSON.parse(await fs.readFile(path.join(source, 'latest-assets.json'), 'utf8'));
    requireValue(data.schema === 1 && data.protocol === 'quality_latest_family_results_v1', 'Unexpected current-results schema');
    requireValue(/^[a-f0-9]{40}$/.test(data.sourceRevision), 'Missing current evidence source revision');
    requireValue(data.scope.historicalGallery === 'historical.html', 'Current results must retain the historical gallery');
    requireValue(isDeepStrictEqual(data.silhouetteLimits, historical.silhouetteLimits), 'Current silhouette limits changed');
    requireValue(data.families.length === 12 && new Set(data.families.map(x => x.id)).size === 12 && isDeepStrictEqual(new Set(data.families.map(x => x.id)), new Set(historical.families.map(x => x.id))), 'Expected the same twelve family identities');
    requireValue(manifest.protocol === 'quality_latest_family_assets_v1' && Array.isArray(manifest.assets), 'Unexpected current asset manifest');
    const assets = new Map(manifest.assets.map(x => [x.path, x]));
    requireValue(assets.size === manifest.assets.length, 'Duplicate current image identity');
    const evidence = new Map();
    async function loadEvidence(binding, family, displayed) {
        requireValue(/^docs\/[a-zA-Z0-9/_-]+\.json$/.test(binding.path) && digest.test(binding.sha256), 'Invalid retained evidence binding');
        if (!evidence.has(binding.path)) {
            const text = (await fs.readFile(path.join(root, binding.path), 'utf8')).replace(/\r\n/g, '\n');
            evidence.set(binding.path, { sha256: hash(text), value: JSON.parse(text) });
        }
        const record = evidence.get(binding.path);
        requireValue(record.sha256 === binding.sha256, 'Retained evidence changed: ' + binding.path);
        requireValue(Array.isArray(binding.assertions) && binding.assertions.length > 0, 'Missing retained evidence assertions');
        for (const assertion of binding.assertions) {
            requireValue(isDeepStrictEqual(pointerValue(record.value, assertion.pointer), assertion.expected), 'Retained evidence assertion changed: ' + binding.path + assertion.pointer);
            for (const display of assertion.display || []) {
                requireValue(['identity', 'boolean_verdict'].includes(display.mapping), 'Unknown displayed evidence mapping');
                if (display.mapping === 'boolean_verdict') requireValue(typeof assertion.expected === 'boolean', 'Verdict mapping requires a recorded boolean');
                const expected = display.mapping === 'identity' ? assertion.expected : (assertion.expected ? 'passed' : 'failed');
                requireValue(isDeepStrictEqual(pointerValue(family, display.pointer), expected), 'Displayed value differs from retained evidence: ' + family.id + display.pointer);
                displayed.add(display.pointer);
            }
        }
        return record;
    }
    for (const family of data.families) {
        requireValue(digest.test(family.geometryHash) && digest.test(family.sourceGeometryHash), 'Invalid current mesh identity: ' + family.id);
        requireValue(gates.every(gate => states.has(family.verdicts[gate])), 'Invalid independent verdict: ' + family.id);
        requireValue(Array.isArray(family.evidenceBindings) && family.evidenceBindings.length > 0, 'Missing current result binding: ' + family.id);
        const displayed = new Set();
        for (const binding of family.evidenceBindings) await loadEvidence(binding, family, displayed);
        const requiredDisplay = ['/geometryHash', '/sourceGeometryHash', '/verdicts/boundary', '/verdicts/topology', '/verdicts/editability'];
        for (const key of ['meanDistance', 'normalP95']) if (family.surface[key] !== undefined) requiredDisplay.push('/surface/' + key);
        for (let index = 0; index < family.viewMetrics.length; index++) for (const key of ['area_iou', 'boundary_iou', 'signed_distance_loss', 'passed']) requiredDisplay.push('/viewMetrics/' + index + '/' + key);
        requireValue(requiredDisplay.every(pointer => displayed.has(pointer)), 'Current display lacks exact scalar/gate evidence bindings: ' + family.id);
        if (!displayed.has('/verdicts/surfaceEngineering')) {
            const limits = family.surface.engineeringLimits;
            requireValue(limits && Number.isFinite(limits.meanDistanceMax) && Number.isFinite(limits.normalP95Max) && displayed.has('/surface/engineeringLimits/meanDistanceMax') && displayed.has('/surface/engineeringLimits/normalP95Max'), 'Derived engineering verdict lacks frozen source limits');
            const passed = family.surface.meanDistance <= limits.meanDistanceMax && family.surface.normalP95 <= limits.normalP95Max;
            requireValue(family.verdicts.surfaceEngineering === (passed ? 'passed' : 'failed'), 'Derived source engineering verdict contradicts its bound raw values/limits');
        }
        requireValue(family.evidenceBindings.some(binding => binding.assertions.some(assertion => assertion.expected === family.geometryHash)), 'Current body not bound to evidence: ' + family.id);
        requireValue(family.viewMetrics.length === 5 && isDeepStrictEqual(new Set(family.viewMetrics.map(x => x.view)), new Set(views)), 'Missing original view metrics: ' + family.id);
        for (const metric of family.viewMetrics) {
            requireValue([metric.area_iou, metric.boundary_iou, metric.signed_distance_loss].every(Number.isFinite), 'Invalid current view metric');
            const passed = metric.area_iou >= data.silhouetteLimits.area && metric.boundary_iou >= data.silhouetteLimits.boundary && metric.signed_distance_loss <= data.silhouetteLimits.distance;
            requireValue(passed === metric.passed, 'Current view verdict contradicts frozen limits: ' + family.id);
        }
        requireValue((family.verdicts.silhouette === 'passed') === family.viewMetrics.every(x => x.passed), 'Current silhouette summary contradicts original views');
        if (family.verdicts.aggregate === 'passed') {
            requireValue(gates.every(gate => family.verdicts[gate] === 'passed'), 'Aggregate pass over an independent gap: ' + family.id);
            requireValue(family.id === 'smooth_vase' && family.evidenceBindings.some(binding => binding.assertions.some(assertion => assertion.pointer === '/accepted' && assertion.expected === true)), 'Aggregate acceptance lacks its retained authored contract');
        }
        if (family.id !== 'smooth_vase') requireValue(family.verdicts.artistSurface === 'unqualified' && family.verdicts.aggregate === 'unqualified', 'Undefined artist criteria were promoted: ' + family.id);
        const slots = new Set();
        for (const image of family.images) {
            requireValue(views.includes(image.view) && ['reference', 'candidate'].includes(image.role), 'Invalid current image role/view');
            requireValue(image.geometryHash === (image.role === 'candidate' ? family.geometryHash : family.sourceGeometryHash), 'Current image depicts a different body: ' + family.id);
            const slot = image.role + '/' + image.view + '/' + image.style;
            requireValue(!slots.has(slot), 'Duplicate current image slot'); slots.add(slot);
            const asset = assets.get(image.path);
            requireValue(asset && asset.geometryHash === image.geometryHash && asset.role === image.role && asset.style === image.style && asset.view === image.view, 'Image manifest contradicts current body or pass');
        }
        requireValue(family.evidence.every(link => /^https:\/\/github\.com\/doublemover\/blendslop\/(blob|tree)\/[a-f0-9]{40}\//.test(link.url)), 'Evidence links must retain a source commit');
    }
    for (const asset of manifest.assets) {
        requireValue(/^assets\/[a-z_]+-latest-[a-z0-9_-]+\.png$/.test(asset.path) && /^docs\/[a-zA-Z0-9/_-]+\.png$/.test(asset.source), 'Invalid current asset path');
        requireValue([asset.sha256, asset.original_sha256, asset.idat_sha256, asset.geometryHash].every(x => digest.test(x)) && asset.pixel_payload_preserved === true && Number.isInteger(asset.bytes) && asset.bytes > 0, 'Invalid current pixel identity');
        requireValue(data.families.some(family => family.images.some(image => image.path === asset.path)), 'Unreferenced current asset');
        requireValue(hash(await fs.readFile(path.join(root, asset.source))) === asset.original_sha256, 'Original retained PNG changed: ' + asset.source);
        const binding = asset.imageBinding, producer = evidence.get(binding?.evidencePath);
        requireValue(producer && pointerValue(producer.value, binding.pointer) === asset.original_sha256, 'Current PNG lacks its exact retained producer hash: ' + asset.path);
    }
    return { data, assets: manifest.assets };
}
