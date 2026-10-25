@echo off
rem ============================================================
rem  Double-click  -> scan every sheet in .\scan_grids
rem                  results in .\cards as Card_grid1_card1.png ...
rem                  used sheets move to .\scanned
rem
rem  Drag & drop a file onto this .bat -> scan just that file
rem  Or in a terminal:  scan.bat sheet.png
rem ============================================================
setlocal
cd /d "%~dp0"

if not "%~1"=="" goto scanfile

if not exist "scan_grids" mkdir "scan_grids"
if not exist "cards" mkdir "cards"

python scan.py scan_grids -o cards --move-done scanned
set "code=%errorlevel%"
echo.
if "%code%"=="2" (
    echo No sheets found - drop your scans into the "scan_grids" folder,
    echo then double-click scan.bat again.
) else if "%code%"=="0" (
    echo Finished. Cards are in the "cards" folder.
) else (
    echo Something failed - check the messages above.
)
pause
exit /b

:scanfile
python scan.py %* -o cards
echo.
pause
exit /b
