# GitHub 仓库更新指南

## 日常更新

在仓库目录打开 PowerShell：

```powershell
git status
git pull --rebase
git add -A
git commit -m "简要说明本次修改"
git push
```

## 推荐的更新顺序

1. 将真实配置替换到 `configs/`，每个正式实验保留一个独立 YAML 文件。
2. 将固定的数据划分清单放入 `data/splits/`，不要只保存随机种子。
3. 将处理后数据上传到 Zenodo 等数据仓库，不要把大文件直接提交到 Git。
4. 在 `data/metadata/` 更新数据清单和字段字典，并在 `data/checksums/` 记录校验值。
5. 用真实实现替换 `src/` 和 `scripts/` 中的 demo 文件。
6. 把表格和图片的源数据保存为 CSV、JSON 或其他开放格式。
7. 更新 README、Data Availability、Code Availability、CITATION.cff 和 CHANGELOG。

## 建立论文对应版本

```powershell
git status
git tag -a v1.0.0-manuscript -m "Reproducibility release for the revised manuscript"
git push origin v1.0.0-manuscript
```

随后在 GitHub Releases 中基于该标签创建 Release，并将该版本归档至 Zenodo。获得 DOI 后，再把 DOI 写回论文和本仓库。

## 大文件注意事项

- 不要提交原始 RF/IQ 大文件、模型检查点或包含敏感信息的本地配置。
- 优先将大数据上传到 Zenodo、Figshare 或机构仓库，然后在本仓库记录 DOI、文件清单和下载说明。
- 提交前运行 `git status` 和 `git diff --cached`，检查是否误传密钥、账号、邮箱或本地绝对路径。
