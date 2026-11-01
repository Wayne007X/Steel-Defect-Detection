# Val 选型报告：Test 尚未开封

NEU-DET was used in previous exploratory experiments. New similarity groups do not establish actual steel-plate or batch identity. This is a rebuilt internal evaluation, not an external untouched blind test.

Selection SHA256: `76faa9d462bf4dd341c98b75d2ae901fae64bb2815fe176719bf36e14d90ea31`

## research
配置 AC_t1_native；共享阈值 0.5；三种子平均 AP50=0.758375、AP95=0.426979。

## balanced
配置 AC_t1_standard；共享阈值 0.35；三种子平均 AP50=0.739962、AP95=0.399973。

## recall
配置 AC_t1_adaptive；共享阈值 0.35；三种子平均 AP50=0.753274、AP95=0.407437。

晋级门槛通过：True
所有 108 个配置/种子条目和 9 档阈值结果完整保留在 grid.json；无 Test 结果参与。