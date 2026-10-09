#!/usr/bin/env node
/** Focused adverse-data checks for current body and acceptance labels. */
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { validateLatest } from './validate-current-demo.mjs';
const root = process.cwd();
const historical = JSON.parse(await fs.readFile(path.join(root, 'site/data.json'), 'utf8'));
const data = JSON.parse(await fs.readFile(path.join(root, 'site/latest.json'), 'utf8'));
const manifest = JSON.parse(await fs.readFile(path.join(root, 'site/latest-assets.json'), 'utf8'));
const scratch = await fs.mkdtemp(path.join(root, 'temp/current-demo-check-'));
async function check(change, expected) {
    const row = structuredClone(data), assets = structuredClone(manifest);
    change(row, assets);
    await fs.writeFile(path.join(scratch, 'latest.json'), JSON.stringify(row));
    await fs.writeFile(path.join(scratch, 'latest-assets.json'), JSON.stringify(assets));
    if (expected) await assert.rejects(validateLatest(root, scratch, historical), expected);
    else assert.equal((await validateLatest(root, scratch, historical)).data.families.length, 12);
}
await check(() => {});
await check(row => { row.families[0].geometryHash = 'f'.repeat(64); }, /body not bound|Displayed value differs/);
await check(row => { row.families[0].evidenceBindings[0].sha256 = 'f'.repeat(64); }, /Retained evidence changed/);
await check(row => { row.families[0].viewMetrics[0].boundary_iou += 0.00001; }, /Displayed value differs/);
await check(row => { row.families[0].surface.meanDistance = 0; }, /Displayed value differs/);
await check(row => { const family = row.families.find(x => x.id !== 'smooth_vase'); family.verdicts.artistSurface = 'passed'; family.verdicts.aggregate = 'passed'; }, /Aggregate acceptance lacks|Undefined artist criteria/);
await check(row => { const family = row.families.find(x => x.images.length); family.images[0].geometryHash = 'f'.repeat(64); }, /different body/);
await check(row => { row.families[1].id = row.families[0].id; }, /same twelve family/);
await check((row, assets) => { assets.assets[0].original_sha256 = 'f'.repeat(64); }, /Original retained PNG changed/);
console.log('Current demo: retained baseline and eight adverse body/evidence/metric/acceptance/pixel-identity checks passed.');
