from __future__ import annotations
import gc
import itertools
from pathlib import Path
import shutil
import numpy as np
from .core import require,load,write,sha,digest,utc,Budget,configure_runtime
from .training import check_environment


def configurations(p):
    result=[]
    for family,tta,profile in itertools.product(p.config['evaluation']['families'],[False,True],p.config['evaluation']['nms_profiles']):
        result.append(dict(id=f'{family}_t{int(tta)}_{profile}',number=len(result),family=family,tta=tta,profile=profile,
                           cost=len(family)*(3 if tta else 1)))
    return result


def cache_tag(family,seed,tta):
    return f"{family}_seed{seed}_{'tta' if tta else 'plain'}"


def records_for(p,jobs,config,seed,split):
    from .prediction import native_capture,combine
    captures=[native_capture(p,p.file(jobs[f'{f}_seed{seed}']['best']),split,cache_tag(f,seed,config['tta']),config['tta']) for f in config['family']]
    require(all(len(v)==len(captures[0]) for v in captures),'Capture image counts differ')
    return [combine(parts,p.config['evaluation']['nms_profiles'][config['profile']]) for parts in zip(*captures)]


def aggregate(records):
    return {key:float(np.mean([r[key] for r in records])) for key in ['AP50','AP95']}


def choose(p,rows):
    configs=configurations(p);groups={}
    for c in configs:
        selected=[r for r in rows if r['config_id']==c['id']]
        require(sorted(r['seed'] for r in selected)==sorted(p.config['seeds']),'Missing/repeated Val seed')
        means=aggregate(selected)
        points=[]
        for i,t in enumerate(p.config['evaluation']['thresholds']):
            points.append(dict(threshold=t,**{k:float(np.mean([r['points'][i][k] for r in selected])) for k in ['precision','recall','F1']}))
        groups[c['id']]=dict(config=c,**means,class_AP50=np.mean([r['class_AP50'] for r in selected],axis=0).tolist(),
                             AP50_std=float(np.std([r['AP50'] for r in selected],ddof=1)),points=points)
    base=groups['B_t0_native'];g=p.config['promotion'];eligible=[]
    for v in groups.values():
        if (v['AP50']>=base['AP50']+g['AP50_gain'] and v['AP95']>=base['AP95']-g['AP95_max_loss'] and
            min(np.array(v['class_AP50'])-base['class_AP50'])>=-g['class_AP50_max_loss']):eligible.append(v)
    if eligible:
        best_ap=max(v['AP50'] for v in eligible)
        research=min([v for v in eligible if v['AP50']>=best_ap-g['AP50_tie']],
                     key=lambda v:(-v['AP95'],v['config']['cost'],v['config']['number']))
    else:research=base
    all_points=[(v,point) for v in groups.values() for point in v['points']]
    balanced,balanced_point=min(all_points,key=lambda pair:(-pair[1]['F1'],pair[0]['config']['cost'],-pair[1]['threshold'],pair[0]['config']['number']))
    limits=p.config['recall_constraints']
    recall_pool=[pair for pair in all_points if pair[1]['precision']>=balanced_point['precision']-limits['precision_loss'] and
                  pair[1]['F1']>=balanced_point['F1']-limits['F1_loss']]
    recall,recall_point=min(recall_pool,key=lambda pair:(-pair[1]['recall'],-pair[1]['F1'],pair[0]['config']['cost'],-pair[1]['threshold'],pair[0]['config']['number']))
    research_point=max(research['points'],key=lambda r:(r['F1'],r['threshold']))
    profiles={name:dict(config=value['config'],threshold=point['threshold'],val_AP50=value['AP50'],val_AP95=value['AP95'],
                        val_workpoint=point) for name,value,point in [('research',research,research_point),('balanced',balanced,balanced_point),('recall',recall,recall_point)]}
    allowed=[v for v in [base,*eligible] if len(v['config']['family'])==1 and not v['config']['tta'] and v['config']['profile']=='native']
    single=min(allowed,key=lambda v:(-v['AP50'],-v['AP95'],v['config']['number']))
    prereg=['A_t0_native','B_t0_native','C_t0_native','D_t0_native','AC_t0_native','AC_t1_native','AC_t1_standard','AC_t1_adaptive']
    test_rows=[dict(label=name,config=groups[name]['config'],threshold=.25) for name in prereg]
    for name,profile in profiles.items():
        key=(profile['config']['id'],profile['threshold'])
        if not any((r['config']['id'],r['threshold'])==key for r in test_rows):
            test_rows.append(dict(label='selected_'+name,config=profile['config'],threshold=profile['threshold']))
    return dict(profiles=profiles,best_single_family=single['config']['family'],promoted=bool(eligible),
                baseline=base,aggregated=list(groups.values()),test_rows=test_rows,
                statement='Three-seed means; seed42 deployment example is fixed, not the best test seed.')


def select_val(p):
    jobs=p.completed_jobs();configure_runtime(p);check_environment(p)
    from .prediction import native_capture,measure
    for family in p.config['families']:
        for seed in p.config['seeds']:
            for tta in [False,True]:
                native_capture(p,p.file(jobs[f'{family}_seed{seed}']['best']),'val',cache_tag(family,seed,tta),tta)
    budget=Budget(p);rows=[]
    with budget.job('Val finite-grid selection'):
        for c in configurations(p):
            for seed in p.config['seeds']:
                budget.check()
                records=records_for(p,jobs,c,seed,'val')
                rows.append(dict(config_id=c['id'],seed=seed,**measure(records,p.config['names'],p.config['evaluation']['thresholds'])))
        grid=dict(split_freeze=p.split_freeze(full=False)['id'],jobs={k:r['best_sha256'] for k,r in jobs.items()},code=p.code_hashes(),results=rows)
        write(p.file('val/grid.json'),grid)
        selection=choose(p,rows);selection.update(grid_sha256=sha(p.file('val/grid.json')),split_freeze=grid['split_freeze'])
        selection['id']=digest(selection);write(p.file('val/selection.json'),selection)
    lines=['# Val 选型报告：Test 尚未开封','',p.config['disclosure'],'',f"Selection SHA256: `{selection['id']}`",'']
    for name,r in selection['profiles'].items():
        lines.extend([f'## {name}',f"配置 {r['config']['id']}；共享阈值 {r['threshold']}；三种子平均 AP50={r['val_AP50']:.6f}、AP95={r['val_AP95']:.6f}。",''])
    lines+=['晋级门槛通过：'+str(selection['promoted']), '所有 108 个配置/种子条目和 9 档阈值结果完整保留在 grid.json；无 Test 结果参与。']
    p.file('val/Val选型报告.md').write_text('\n'.join(lines),encoding='utf-8')
    return selection


def get_selection(p):
    jobs=p.completed_jobs();check_environment(p)
    s=load(p.file('val/selection.json'));grid=load(p.file('val/grid.json'))
    require(s['id']==digest({k:v for k,v in s.items() if k!='id'}),'Val selection changed')
    require(s['grid_sha256']==sha(p.file('val/grid.json')) and grid['code']==p.code_hashes(),'Val selection code/results changed')
    require(grid['jobs']=={k:r['best_sha256'] for k,r in jobs.items()},'Models changed since Val selection')
    return jobs,s


def export_verify(p):
    configure_runtime(p);jobs,selection=get_selection(p)
    from .prediction import actual_capture,measure
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(2);budget=Budget(p);reports=[]
    with budget.job('Val real-runtime and ONNX verification'):
        seen={}
        for name,chosen in selection['profiles'].items():
            budget.check();c=chosen['config'];targets=records_for(p,jobs,c,42,'val')
            key=c['id']
            if key not in seen:
                weights=[p.file(jobs[f'{f}_seed42']['best']) for f in c['family']]
                real,latency=actual_capture(p,weights,targets,c['tta'],p.config['evaluation']['nms_profiles'][c['profile']],'cuda:0','val',budget)
                expected=measure(targets,p.config['names'],[chosen['threshold']]);observed=measure(real,p.config['names'],[chosen['threshold']])
                require(max(abs(expected[k]-observed[k]) for k in ['AP50','AP95'])<=.002,'Runtime/native parity failed')
                seen[key]=dict(config=c,latency=latency,observed=observed,reference=expected)
            reports.append(dict(kind='gpu_runtime',profile=name,**seen[key]))
        families=set(selection['profiles']['balanced']['config']['family'])|{selection['best_single_family']}
        exports={}
        for family in sorted(families):
            budget.check();source=p.file(jobs[f'{family}_seed42']['best']);destination=p.file(f'exports/{family}_seed42.onnx')
            receipt=destination.with_suffix('.provenance.json')
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                require(receipt.exists(),'Incomplete ONNX export: inspect artifact before retrying')
                previous=load(receipt)
                require(previous['source_sha256']==sha(source) and previous['onnx_sha256']==sha(destination),'ONNX export provenance changed')
            else:
                options={k:v for k,v in p.config['onnx'].items() if k!='AP_tolerance'}
                result=YOLO(str(source)).export(**options);shutil.copy2(result,destination)
                write(receipt,dict(source=str(source.relative_to(p.root)),source_sha256=sha(source),onnx_sha256=sha(destination),options=options))
            exports[family]=destination
        cases=[('best_single',dict(family=selection['best_single_family'],tta=False,profile='native')),
               ('balanced_no_tta_variant',{**selection['profiles']['balanced']['config'],'tta':False})]
        for name,c in cases:
            budget.check();targets=records_for(p,jobs,c,42,'val')
            actual,latency=actual_capture(p,[exports[f] for f in c['family']],targets,False,
                           p.config['evaluation']['nms_profiles'][c['profile']],'cpu','val',budget)
            reference=measure(targets,p.config['names'],[.25]);observed=measure(actual,p.config['names'],[.25])
            delta={k:observed[k]-reference[k] for k in ['AP50','AP95']}
            reports.append(dict(kind='onnx',profile=name,config=c,reference=reference,observed=observed,delta=delta,latency=latency,
                                deployable=max(abs(v) for v in delta.values())<=p.config['onnx']['AP_tolerance'],
                                matches_balanced_gpu_tta=not selection['profiles']['balanced']['config']['tta'] if name.startswith('balanced') else None,
                                weights=[dict(path=str(exports[f].relative_to(p.root)),sha256=sha(exports[f])) for f in c['family']]))
        result=dict(selection_id=selection['id'],profiles=reports,all_gpu_parity_passed=True,
                    disclosure='ONNX does not perform TTA. A no-TTA variant is explicitly validated; it is not claimed equivalent to a TTA-enabled selected GPU profile.')
        write(p.file('val/runtime_export_verification.json'),result)
    return result


def freeze_final(p):
    jobs,selection=get_selection(p);split=p.split_freeze()
    report=load(p.file('val/runtime_export_verification.json'))
    require(report['selection_id']==selection['id'] and report['all_gpu_parity_passed'],'Runtime verification missing or stale')
    files=[*p.file('val').glob('*.json'),*p.file('val').glob('*.md'),*p.file('exports').glob('*.onnx'),*p.file('exports').glob('*.json')]
    for name,r in jobs.items():
        files.extend([p.file(r['best']),*[p.file(f'runs/{name}/{file}') for file in ['completion.json','initialization.json','args.yaml','results.csv','ap50_best.json']]])
    files.extend(p.file('cache/val').glob('*/complete.json'));files.extend(p.file('cache/val').glob('*/records.npy'))
    record=dict(split_freeze=split['id'],selection=selection['id'],files={**split['files'],**p.fingerprint(files)},
                code=p.code_hashes(),test_rows=selection['test_rows'],seeds=p.config['seeds'],frozen_at=utc(),
                deployment_seed=42,selected_profiles=selection['profiles'],disclosure=p.config['disclosure'])
    record['id']=digest(record);write(p.file('freezes/final.json'),record,exclusive=True)
    path=p.file('freezes/最终冻结报告.md')
    path.write_text('# 最终冻结报告：等待 Test 开封确认\n\n'+p.config['disclosure']+'\n\nFinal SHA256: `'+record['id']+'`\n\n'+
                    '请核对 Val 选型报告、完整模型/代码/数据哈希、ONNX 的无 TTA 限制及全部预登记测试行。\n'+
                    '未经明确确认不得运行 approve-test。开封后不再根据 Test 改模型、阈值或种子。\n\n'+
                    '\n'.join(f"- {r['label']}: {r['config']['id']}, threshold={r['threshold']}" for r in record['test_rows']),encoding='utf-8')
    return record


def evaluate_test(p):
    configure_runtime(p);freeze=p.test_guard();check_environment(p);jobs=p.completed_jobs();budget=Budget(p);budget.check()
    from .prediction import measure
    result_path=p.file('test/final_results.json')
    if result_path.exists():
        r=load(result_path);require(r['freeze_id']==freeze['id'],'Test report belongs to another freeze');return r
    started=p.file('test/evaluation_started.json')
    if not started.exists():write(started,dict(freeze_id=freeze['id'],started_at=utc(),test_list_sha256=sha(p.file('dataset/test.txt'))),exclusive=True)
    else:require(load(started)['freeze_id']==freeze['id'],'Cannot reuse an opened Test with another freeze')
    results=[]
    for row in freeze['test_rows']:
        seeds=[]
        for seed in freeze['seeds']:
            budget.check();records=records_for(p,jobs,row['config'],seed,'test')
            with budget.job('Test metrics:'+row['label']+f':{seed}'):
                seeds.append(dict(seed=seed,**measure(records,p.config['names'],[row['threshold']])))
        result=dict(label=row['label'],config=row['config'],threshold=row['threshold'],seeds=seeds,
                    mean=aggregate(seeds),std={k:float(np.std([r[k] for r in seeds],ddof=1)) for k in ['AP50','AP95']})
        for metric in ['precision','recall','F1']:
            values=[r['points'][0][metric] for r in seeds]
            result['mean'][metric]=float(np.mean(values));result['std'][metric]=float(np.std(values,ddof=1))
        result['per_class_summary']=[dict(name=name,**{key+suffix:float(function([r[key][c] for r in seeds]))
                     for key in ['class_AP50','class_AP95'] for suffix,function in [('_mean',np.mean),('_std',lambda v:np.std(v,ddof=1))]})
                     for c,name in enumerate(p.config['names'])]
        results.append(result)
        write(p.file('test/in_progress.json'),dict(freeze_id=freeze['id'],results=results))
    r=dict(freeze_id=freeze['id'],disclosure=p.config['disclosure'],finished_at=utc(),results=results,
           class_names=p.config['names'],selected_profiles=freeze['selected_profiles'],
           selection_policy='All rows and thresholds fixed on Val before release; no Test ranking is used to change the selected profiles.')
    write(result_path,r,exclusive=True)
    lines=['# NEU-DET 新固定协议统一 Test 报告','',p.config['disclosure'],'',f"Final freeze: `{freeze['id']}`",'']
    lines+=['## 预先在 Val 冻结的选择',*[f"- {name}: {v['config']['id']}, threshold={v['threshold']}" for name,v in freeze['selected_profiles'].items()],
            '这些选择不随下面的 Test 分数改变。','']
    for row in results:
        lines.extend([f"## {row['label']}",f"{row['config']['id']}，固定阈值 {row['threshold']}",
                      f"AP50 均值 {row['mean']['AP50']:.6f}、样本标准差 {row['std']['AP50']:.6f}；AP95 均值 {row['mean']['AP95']:.6f}、样本标准差 {row['std']['AP95']:.6f}。"])
        lines.append('工作点三种子均值/样本标准差：'+', '.join(f"{k}={row['mean'][k]:.6f}/{row['std'][k]:.6f}" for k in ['precision','recall','F1']))
        for seed in row['seeds']:
            point=seed['points'][0];lines.append(f"- seed {seed['seed']}: AP50={seed['AP50']:.6f}, AP95={seed['AP95']:.6f}, P={point['precision']:.6f}, R={point['recall']:.6f}, F1={point['F1']:.6f}, TP/FP/FN={point['TP']}/{point['FP']}/{point['FN']}")
        lines.extend(f"- {v['name']}: AP50 均值/标准差={v['class_AP50_mean']:.6f}/{v['class_AP50_std']:.6f}, AP95 均值/标准差={v['class_AP95_mean']:.6f}/{v['class_AP95_std']:.6f}" for v in row['per_class_summary'])
        lines.append('')
    lines+=['逐类 AP50/AP95 和工作点逐类计数完整保存在 final_results.json，未仅保留最高分。',
            '此结果不等同于新来源外部盲测，不能证明跨产线、相机和照明的生产泛化能力。']
    p.file('test/Test统一评估报告.md').write_text('\n'.join(lines),encoding='utf-8')
    return r
