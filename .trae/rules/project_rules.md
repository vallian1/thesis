# 项目规则

## LaTeX 编译器设置

本项目使用 TeX Live 2026 编译器，路径为：
```
E:\texlive\2026\bin\windows\
```

### 编译命令

```powershell
# 使用完整路径编译
E:\texlive\2026\bin\windows\xelatex.exe -interaction=nonstopmode xdupgthesis_template_lc.tex

# 处理参考文献
E:\texlive\2026\bin\windows\biber.exe xdupgthesis_template_lc

# 完整编译流程
E:\texlive\2026\bin\windows\xelatex.exe xdupgthesis_template_lc.tex
E:\texlive\2026\bin\windows\biber.exe xdupgthesis_template_lc
E:\texlive\2026\bin\windows\xelatex.exe xdupgthesis_template_lc.tex
E:\texlive\2026\bin\windows\xelatex.exe xdupgthesis_template_lc.tex
```

### 环境变量设置（PowerShell）

```powershell
$env:PATH = "E:\texlive\2026\bin\windows;$env:PATH"
```
