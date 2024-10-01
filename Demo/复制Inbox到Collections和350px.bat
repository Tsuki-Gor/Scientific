rem 将“C:\Users\mircocrift\Documents\GitHub\Scientific\Demo\Inbox”路径下的文件都复制一份到目录“C:\Users\mircocrift\Documents\GitHub\Scientific\Demo\Collections\Original”。之后将“C:\Users\mircocrift\Documents\GitHub\Scientific\Demo\Inbox”路径下的文件在末尾添加350px并移动到路径“C:\Users\mircocrift\Documents\GitHub\Scientific\Demo\Collections\350px”下

@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

rem 定义目录路径
set "inbox=C:\Users\mircocrift\Documents\GitHub\Scientific\Demo\Inbox"
set "original=C:\Users\mircocrift\Documents\GitHub\Scientific\Demo\Collections\Original"
set "destination=C:\Users\mircocrift\Documents\GitHub\Scientific\Demo\Collections\350px"

rem 创建目标目录，如果不存在
if not exist "%original%" (
    mkdir "%original%"
)
if not exist "%destination%" (
    mkdir "%destination%"
)

rem 复制文件到 Original 目录
xcopy "%inbox%\*" "%original%\" /s /e /y

rem 重命名并移动文件到 350px 目录
for %%f in ("%inbox%\*") do (
    if not "%%~ff"=="%inbox%\*" (
        set "filename=%%~nf"
        set "extension=%%~xf"
        set "newname=!filename!350px!extension!"
        move "%%~ff" "%destination%\!newname!"
    )
)

echo 所有操作已完成！
pause