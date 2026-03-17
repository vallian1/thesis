# 设置 TeX Live 2026 为默认编译器
$texlive2023 = "D:\texlive\2023\bin\windows"
$texlive2026 = "E:\texlive\2026\bin\windows"

# 获取当前用户 PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

# 移除 2023 路径
$newPath = ($userPath -split ';' | Where-Object { $_ -ne $texlive2023 }) -join ';'

# 确保 2026 路径存在
if ($newPath -notcontains $texlive2026) {
    $newPath = $texlive2026 + ";" + $newPath
}

# 设置新的 PATH
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")

Write-Host "已更新用户环境变量 PATH"
Write-Host "请重启 Trae IDE 或电脑使更改生效"
