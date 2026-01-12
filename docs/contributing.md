# 贡献指南

感谢你对 NeuroHear 的关注！欢迎各种形式的贡献。

## 如何贡献

### 报告 Bug

在 [Issues](https://github.com/neurohear/neurohear/issues) 提交 Bug 报告，请包含：
- 问题描述
- 复现步骤
- 期望行为
- 实际行为
- 环境信息（Python 版本、PyTorch 版本、操作系统）

### 功能建议

在 [Discussions](https://github.com/neurohear/neurohear/discussions) 讨论新功能想法。

### 提交代码

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m "Add your feature"`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

## 开发环境

```bash
# 克隆你的 fork
git clone https://github.com/YOUR_USERNAME/neurohear.git
cd neurohear

# 安装开发依赖
uv sync --extra dev

# 运行测试
uv run pytest

# 代码格式化
uv run ruff format .
uv run ruff check --fix .
```

## 代码规范

- 使用 [ruff](https://github.com/astral-sh/ruff) 进行代码格式化和 lint
- 遵循 PEP 8 风格指南
- 为公共 API 编写 docstring
- 添加适当的类型注解

## 测试

- 为新功能添加测试
- 确保所有测试通过：`uv run pytest`
- 保持测试覆盖率

## 文档

- 更新相关文档
- 为新功能添加示例
- 保持 README 和 docs/ 同步

## 行为准则

请友善、尊重地与他人交流。我们致力于维护一个开放、包容的社区环境。
