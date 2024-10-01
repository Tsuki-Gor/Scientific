@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "prefix=https://scientific-github.5208818.xyz/Demo/Collections/350px/"

for /f "delims=" %%a in ('powershell -command "Get-Clipboard"') do (
    if not "%%a"=="" (
        set "line=!prefix!%%a"
        rem 将空格替换为 %20
        set "line=!line: =%%20!"
        echo !line!
    )
)

pause