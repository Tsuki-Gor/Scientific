@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 设置变量
set "inputDir=C:\Users\mircocrift\Documents\GitHub\Scientific\Demo\Collections\350px\"
set "searchString=padding: 20px;"
set "replaceString=padding: 20px 350px;"

REM 从剪贴板获取文件名列表并存入临时文件
powershell -command "Get-Clipboard" > clipboard_temp.txt

REM 检查临时文件是否存在以及是否有内容
if not exist clipboard_temp.txt (
    echo 错误: 未能获取剪贴板内容。
    goto :end
)

REM 计算文件总数
for /f %%A in ('find /c /v "" ^< clipboard_temp.txt') do set "fileCount=%%A"

echo 当前剪贴板内容:
echo --------------------------------
set "fileIndex=0"
for /f "delims=" %%i in (clipboard_temp.txt) do (
    echo 文件 %%i
    set /a fileIndex+=1
)
echo --------------------------------
echo 一共有 !fileCount! 个文件需要处理
echo --------------------------------

pause

REM 初始化处理计数和序号
set "processedCount=0"
set "sequence=0"

REM 从临时文件读取文件名列表并处理
for /f "delims=" %%i in (clipboard_temp.txt) do (
    set /a sequence+=1
    set "inputFile=!inputDir!%%i"
    echo ************************
    echo 正在处理文件 !sequence! / !fileCount!: !inputFile!

    REM 执行替换操作，指定编码为 UTF-8 无 BOM 并使用 UNIX 行尾
    powershell -Command ^
      "$lines = Get-Content -Path '!inputFile!' -Encoding UTF8;" ^
      "$search = [regex]::Escape('!searchString!');" ^
      "$match = $lines | Select-String -Pattern $search -List | Select-Object -First 1;" ^
      "if ($match) {" ^
          "$lineNumber = $match.LineNumber;" ^
          "$index = $lineNumber - 1;" ^
          "$start = [Math]::Max(0, $index - 1);" ^
          "$end = [Math]::Min($lines.Count - 1, $index + 2);" ^
          "Write-Host '--- 当前替换位置发生的行号 ---';" ^
          "Write-Host $lineNumber;" ^
          "Write-Host '--- 替换位置的上下四行原始内容 ---';" ^
          "for ($i = $start; $i -le $end; $i++) {" ^
              "Write-Host $lines[$i]" ^
          "}" ^
          "$lines[$index] = $lines[$index] -replace $search, '!replaceString!';" ^
          "$content = $lines -join \"`n\";" ^
          "[System.IO.File]::WriteAllText('!inputFile!', $content, (New-Object System.Text.UTF8Encoding $false))" ^
      "} else {" ^
          "Write-Host '！！！！！！！！！！！未找到搜索字符串！！！！！！！！！！！'" ^
      "}"

    REM 检查 PowerShell 命令是否成功执行
    if !errorlevel! == 0 (
        set /a processedCount+=1
    )

    echo --------------------------------
)

REM 输出处理结果
echo 一共实际处理了 !processedCount! 个文件
echo --------------------------------

REM 删除临时文件
del /f /q clipboard_temp.txt

goto :end

:end
endlocal
pause