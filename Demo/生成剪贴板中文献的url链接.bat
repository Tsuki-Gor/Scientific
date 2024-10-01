@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "prefix=https://scientific-github.5208818.xyz/Demo/Collections/350px/"

for /f "delims=" %%a in ('powershell -command "Get-Clipboard"') do (
    if not "%%a"=="" (
        echo !prefix!%%a
    )
)

pause