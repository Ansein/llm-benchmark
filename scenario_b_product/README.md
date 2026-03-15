# Scenario B Product Workspace

该目录是场景B产品化的独立工作区。

原则：
- 不修改仓库原有实现。
- 所有产品化开发在本目录完成。

目录说明：
- backend/: 场景B相关后端代码副本（可继续重构为API/任务系统）
- frontend/: 前端代码目录（待实现）
- docs/: 产品化相关文档副本

已复制基础文件：
- backend/src/evaluate_scenario_b.py
- backend/src/llm_client.py
- backend/run_evaluation.py
- backend/run_prompt_experiments.py
- backend/run_sensitivity_b.py
- docs/场景B产品化补充功能方案.md
- docs/appendix_prompts.tex
- docs/appendix_d_solver.tex
- docs/appendix_b_sensitivity.tex

建议下一步：
1. 在 backend/src 下新增 app/ 目录，搭建 FastAPI 入口。
2. 把 run_*.py 的 CLI 逻辑抽成 service 函数。
3. 接入 run_id 与结果入库。
