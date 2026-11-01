"""Initial handoff verification. Refuses live compute commands once human review has begun."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from neu_protocol.core import Protocol,require,write,utc,sha


def main():
    root=Path(__file__).resolve().parent;p=Protocol(root)
    env={**os.environ,'PYTHONIOENCODING':'utf-8'}
    def run(arguments):
        return subprocess.run([sys.executable,*arguments],cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    status=run(['run_protocol.py','status']);require(status.returncode==0,'Cannot read initial status')
    before=json.loads(status.stdout)
    require(not before['human_review_imported'] and not before['split_frozen'] and not before['test_authorized'],
            'Initial handoff checks are not permitted after approvals; run standalone unit tests instead.')
    units=run(['-m','unittest','discover','-s','tests','-v'])
    p.file('implementation_tests.log').write_text(units.stdout+'\n'+units.stderr,encoding='utf-8')
    require(units.returncode==0,'Implementation unit tests failed; inspect implementation_tests.log')
    match=re.search(r'Ran (\d+) tests',units.stderr)
    commands=[['train'],['select-val'],['verify-export'],['freeze-final'],['test'],
              ['approve-test','--freeze-sha','unapproved-fixture','--reviewer','guard-check-not-approval'],
              ['propose-split'],['import-review','--file','review/decisions_template.json']]
    guards=[]
    for command in commands:
        result=run(['run_protocol.py',*command])
        require(result.returncode==2 and 'PROTOCOL BLOCKED:' in result.stderr,'Unsafe guard result: '+str(command))
        guards.append(dict(command=command,exit_code=result.returncode,message=result.stderr.strip()))
    after=json.loads(run(['run_protocol.py','status']).stdout)
    require(after==before,'Guard checks mutated experimental state')
    require(not p.file('runs').exists() and not p.file('cache').exists() and not p.file('test').exists(),
            'Guard checks must not create any real model run or prediction cache')
    p.source_integrity()
    official=json.loads(p.file('models/official_provenance.json').read_text(encoding='utf-8'))
    require(official['sha256']==sha(p.file('models/yolov8n_coco.pt')),'Official weight integrity failed')
    audit=json.loads(p.file('audit/summary.json').read_text(encoding='utf-8'))
    report=dict(verified_at=utc(),unit_tests=int(match.group(1)) if match else None,unit_tests_passed=True,
                actual_initial_stage_guard_checks=guards,state=after,source_integrity_passed=True,
                audited_images=audit['images'],candidate_pairs=audit['candidate_pairs'],
                official_weight_sha256=official['sha256'],code=p.code_hashes(),
                real_NEU_training_executed=False,real_Test_predictions_executed=False,
                remaining_validation='Actual 12-task training, Val runtime/export parity and final Test reporting remain gated by the specified human approvals.')
    write(p.file('reports/implementation_verification.json'),report)
    lines=['# 实现检查与当前状态','',f"检查时间（UTC）：{report['verified_at']}",'',
           f"- 隔离单元／集成检查：{report['unit_tests']} 项通过。",
           '- 包含真实固定版本分组 worker、合成图验证数据集形状、AP50 fitness/早停/同分保留、预算、来源链和选择规则检查。',
           f"- 在真实初始项目目录运行 {len(guards)} 个未满足前置条件的阶段命令，均以退出码 2 拒绝，实验状态完全不变。",
           '- 已复核 1800 张原始与规范副本的图片、标签及 XML 校验；官方 COCO 权重 SHA256 匹配。',
           '- 尚未导入人工分组决定，没有新 Train/Val/Test 划分，没有训练任务，没有真实模型预测缓存或 Test 分数。',
           '- 因人工复核门槛尚未满足，真实重训、Val 实际推理/导出一致性与最终 Test 验证尚未执行；不把合成测试当成模型性能结果。','',
           '## 当前应由用户复核','',
           '请查看 review/人工分组复核.md 和两张总览。P001 已确认完全重复、固定同组；P002–P019 需要判断 same / different / uncertain。',
           '完成后才能生成划分审计，待用户确认审计及读取规则后冻结并开始训练。','',
           '## 额外数据口径','',
           '源图片主类别每类 300 张，但其中 123 张含其他类别目标；所有原始标签保留。',
           '三个标签文件各有一行完全重复框，按固定框架内存去重行为，原始 4189 框对应有效 4186 框；两种计数均写入后续划分审计。']
    p.file('reports/实现检查与待复核事项.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(unit_tests=report['unit_tests'],blocked_guards=len(guards),state=after),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
