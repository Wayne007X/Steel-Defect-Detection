# Streamlit 自动唤醒

工作流每 4 小时（北京时间 00:17、04:17、08:17、12:17、16:17、20:17）使用 Chromium 打开公开演示。发现休眠按钮时点击唤醒，等待应用标题及图片上传控件出现，保留浏览器连接 20 秒。最多重试三次；加载失败会让 Actions 标红并保留截图与日志 7 天。无需密钥或本地电脑开机。

在仓库 Actions → Streamlit browser keep-alive → Run workflow 可立即执行。失败通知受个人 GitHub Actions 通知设置控制，建议开启邮件通知。

这属于尽力保活，不保证免费托管的可用性。GitHub 定时任务可能延迟，公开仓库连续 60 天没有仓库活动会自动停用定时工作流；届时需在 Actions 重新启用，并定期检查最近一次成功运行。不要用自动空提交伪造仓库活动。Streamlit 平台故障、资源超限、规则变化及依赖故障也可能导致应用不可用。若必须持续在线，需要迁移到不强制休眠的托管服务。

此检查验证前端可加载，不执行图片推理，也不保证独立后端可用。面试前仍建议上传一张图片完成端到端验证。

依据：
- https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
- https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule
