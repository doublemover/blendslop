"""Focused saved-arch scope, original-owner reuse and physical-edit guards."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_arch_final_qualification as packet
from evaluation.controlled_measurement import measurement_signature
from utils.run_ownership import OwnedRun


def fixture(parent):
    """Real temporary ownership/files; acquisition values are explicitly synthetic."""
    owner = OwnedRun(parent, producer=packet.ORIGINAL_PRODUCER)
    wire = {'program_id': 'frozen-family-concave_arch', 'root_nodes': [{
        'primitive_type': 'polygon_extrusion', 'parameters': {'holes': [],
        'outer': [[-1,-1],[-1,1],[1,1],[1,-1],[.4,-1],[.4,.4],[-.4,.4],[-.4,-1]],
        'height_world': .4, 'x': .02}}], 'metadata': {'historical': 'baseline observations remain historical'}}
    edited = deepcopy(wire); edited['root_nodes'][0]['parameters']['height_world'] *= 1.05
    response = {'control': 'extrusion_depth_world', 'expected_multiplier': 1.05,
        'checks': {'fixed_all_outline_vertices_xy': True, 'unchanged_triangle_inventory': True, 'depth_response': True},
        'passed': True}
    edit = {'control': 'extrusion_depth_world x1.05', 'response': response,
        **dict.fromkeys(('passed','geometry_changed','same_source_pointer','original_mesh_restored','exact_indexed_restoration'), True)}
    contract = {'protocol': 'opaque_linear_alpha_v1', 'hard_mask_threshold': .5,
        'camera': {'projection': 'ORTHO', 'matrix_world': np.eye(4).tolist(), 'ortho_scale': 2.,
            'shift_x': 0., 'shift_y': 0., 'clip_start': .1, 'clip_end': 1000.,
            'resolution': [512,512], 'pixel_aspect': [1.,1.]},
        'renderer': {'engine': 'BLENDER_EEVEE'}, 'sampling': {'taa_render_samples':64,'filter_size':1.5},
        'pixel_convention':'top-left','surface_policy':'opaque','encoding':'linear alpha'}
    history = {'oblique_145_40': 'earlier inspected; excluded current fit'}
    row = {'status':'measured', 'program':wire, 'refined_geometry_hash':packet.CANDIDATE,
        'source_original_geometry_hash':packet.SOURCE, 'source_replayed_geometry_hash':packet.SOURCE,
        'source_exact_oriented_equivalence':True,'fixed_inputs_geometry_stable':True,
        'artist_surface_limits':None,'aggregate_accepted':False,'semantic_edit_restoration':edit,
        'refined_raw_surface': {'protocol':'shared_world_area_sample_to_triangle_v1',
            'sample_count_per_direction':4096,'seed':61007,
            'candidate_geometry_hash':packet.CANDIDATE,'reference_geometry_hash':packet.SOURCE}}
    with owner:
        folder = owner.root/packet.FAMILY
        (folder/'refined').mkdir(parents=True)
        measurements = folder/'source-heldout/oblique_145_40'; measurements.mkdir(parents=True)
        def write(relative, value):
            p = owner.root/relative; p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(value)); owner.register_file(relative,'final_output'); return packet.shared.bind(p)
        manifest = None
        bindings = {'candidate_program':write(packet.FAMILY+'/refined/program.json',wire),
            'semantic_edited_program':write(packet.FAMILY+'/semantic-edited-program.json',edited)}
        for key,name in (('candidate_npz','evaluated-exact.npz'),('candidate_obj','evaluated.obj')):
            p=folder/'refined'/name;p.write_bytes(b'synthetic-body-transport-not-native-evidence')
            owner.register_file(p.relative_to(owner.root),'final_output');bindings[key]=packet.shared.bind(p)
        receipt=write('results.json',{'protocol':'bounded_arch_exterior_checkpoint_v1','status':'measured',
                                    'cases':{packet.FAMILY:row}})
        frozen=write('frozen-workload.json',{'cases':[{'prior_view_exposure':history}]})
        crop=write(packet.FAMILY+'/crop-request.json',{'status':'requested','fit_view':'oblique_35_28_expanded',
                                                     'heldout_views':['oblique_145_40'],'prior_view_exposure':history})
        values=np.zeros((512,512),np.float32);values[10,10:13]=[.499,.5,.501]
        np.save(measurements/'coverage.npy',values,allow_pickle=False)
        (measurements/'alpha.exr').write_bytes(b'synthetic-original-alpha-transport')
        meta={'contract':contract,'contract_sha256':measurement_signature(contract),'geometry_hashes':[packet.SOURCE],
            'geometry_unchanged':True,'exr_sha256':packet.shared.sha(measurements/'alpha.exr')}
        write(packet.FAMILY+'/source-heldout/oblique_145_40/measurement.json',meta)
        for name in ('coverage.npy','alpha.exr'):owner.register_file((measurements/name).relative_to(owner.root),'final_output')
    manifest=packet.shared.bind(owner.root/'run-ownership.json');lease=packet.shared.bind(owner.root/'run-lease.json')
    plan={'case':bindings,'checkpoint_root':str(owner.root),'checkpoint_manifest':manifest,'checkpoint_lease':lease,
        'checkpoint_receipt':receipt,'historical_frozen_workload':frozen,'crop_request':crop,'prior_view_exposure':history}
    reuse={'owner_root':str(owner.root),'manifest':manifest,'lease':lease,'label':'source-heldout/oblique_145_40',
        'files':{name:packet.shared.bind(measurements/name) for name in ('measurement.json','coverage.npy','alpha.exr')}}
    return plan,reuse,contract,values,wire,edited,edit


def scope():
    return {'protocol':packet.PROTOCOL,'family':packet.FAMILY,'views':list(packet.shared.CANONICAL_VIEWS),
        'passes':list(packet.shared.INSPECTION_PASSES),'resolution':[512,512],'native_frames':28,
        'retained_alpha_passes':2,'logical_inspection_passes':30,'fits':0,'raw_comparisons':0,'semantic_transactions':0,
        'qualifier_children':1,'qualifier_timeout_seconds':15,'work_seconds':85,'join_seconds':5,'threads':2,
        'committed_limit_bytes':packet.shared.MEMORY,'rss_limit_bytes':packet.shared.MEMORY,
        'artifact_limit_bytes':268435456,'pass_write_reservation_bytes':8388608,
        'artist_surface_limits':None,'aggregate_accepted':False,'retained_measurement_producer':packet.ORIGINAL_PRODUCER,
        'gates':{'minimum_area_iou':.7,'minimum_boundary_iou':.8,'maximum_signed_distance_loss':.05},
        'case':{'source_geometry_hash':packet.SOURCE,'candidate_geometry_hash':packet.CANDIDATE},
        'reuse':{role:{v:{'label':label} for v,label in rows.items()} for role,rows in packet.REUSE.items()}}


class ArchFinalQualificationTests(unittest.TestCase):
    def test_exact_budget_source_body_and_two_ob145_slots_refuse_scope_expansion(self):
        original=scope();packet.require_scope(original)
        for key,value in (('native_frames',29),('retained_alpha_passes',3),('fits',1),('raw_comparisons',1),
            ('semantic_transactions',1),('qualifier_children',True),('retained_measurement_producer','unknown')):
            changed=deepcopy(original);changed[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'frozen scope'):packet.require_scope(changed)
        changed=deepcopy(original);changed['reuse']['source']['top']=changed['reuse']['source'].pop('oblique_145_40')
        with self.assertRaisesRegex(ValueError,'two retained'):packet.require_scope(changed)
        changed=deepcopy(original);changed['case']['candidate_geometry_hash']='e'*64
        with self.assertRaisesRegex(ValueError,'selected source/body'):packet.require_scope(changed)

    def test_arch_owner_reuse_requires_explicit_producer_and_exact_actual_clips(self):
        with tempfile.TemporaryDirectory() as parent:
            _,reuse,c,values,_,_,_=fixture(parent);settings=packet.shared.non_camera(c)
            row,origin=packet.shared.load_reuse(reuse,packet.SOURCE,settings,c['camera'],
                family=packet.FAMILY,producer=packet.ORIGINAL_PRODUCER)
            np.testing.assert_array_equal(row['coverage'],values)
            self.assertEqual(row['mask'][10,10:13].tolist(),[False,True,True])
            self.assertEqual(origin['owner_root'],reuse['owner_root'])
            with self.assertRaisesRegex(ValueError,'ownership'):
                packet.shared.load_reuse(reuse,packet.SOURCE,settings,family=packet.FAMILY)
            camera=deepcopy(c['camera']);camera['clip_end']=100.
            with self.assertRaisesRegex(ValueError,'mismatch'):
                packet.shared.load_reuse(reuse,packet.SOURCE,settings,camera,
                    family=packet.FAMILY,producer=packet.ORIGINAL_PRODUCER)

    def test_only_original_depth_multiplier_and_all_response_restore_checks_are_admitted(self):
        with tempfile.TemporaryDirectory() as parent:
            _,_,_,_,wire,edited,edit=fixture(parent)
            original=deepcopy((wire,edited,edit));packet.require_existing_edit(wire,edited,edit)
            self.assertEqual((wire,edited,edit),original)
            for change in ('outline','pose','metadata','multiplier','restore','physical_response'):
                e,r=deepcopy(edited),deepcopy(edit)
                if change=='outline':e['root_nodes'][0]['parameters']['outer'][0][0]-=.01
                elif change=='pose':e['root_nodes'][0]['parameters']['x']+=.01
                elif change=='metadata':e['metadata']['historical']='silently changed'
                elif change=='multiplier':r['response']['expected_multiplier']=1.1
                elif change=='restore':r['exact_indexed_restoration']=False
                else:r['response']['checks']['depth_response']=False
                with self.subTest(change=change),self.assertRaises(ValueError):packet.require_existing_edit(wire,e,r)

    def test_original_checkpoint_hashes_history_and_owned_edit_cannot_be_rebound(self):
        with tempfile.TemporaryDirectory() as parent:
            plan,_,_,_,_,_,_=fixture(parent);original=deepcopy(plan)
            row=packet.verify_checkpoint(plan);self.assertEqual(row['refined_geometry_hash'],packet.CANDIDATE)
            self.assertEqual(plan,original)
            changed=deepcopy(plan);changed['prior_view_exposure']['oblique_145_40']='claimed blind'
            with self.assertRaisesRegex(ValueError,'exposure history'):packet.verify_checkpoint(changed)
            edited=Path(plan['case']['semantic_edited_program']['path']);edited.write_bytes(edited.read_bytes()+b' ')
            changed=deepcopy(plan);changed['case']['semantic_edited_program']=packet.shared.bind(edited)
            with self.assertRaisesRegex(ValueError,'owned artifact bytes'):packet.verify_checkpoint(changed)


if __name__=='__main__':unittest.main()
