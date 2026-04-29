@echo off
setlocal EnableDelayedExpansion
title Make Flowchart PDF
cd /d "%~dp0"
echo.
echo  flowchart.html  --^>  flowchart.svg  +  flowchart.pdf
echo.

:: ── Step 1: extract SVG from HTML ────────────────────────────────────────────
:: Note: avoid %d in Python strings — cmd expands it before Python sees it.
::       Use string concatenation instead of %-formatting.
python -c "import re,pathlib;d=pathlib.Path('.');t=d.joinpath('flowchart.html').read_text(encoding='utf-8');m=re.search(r'(<svg[\s\S]*?</svg>)',t);d.joinpath('flowchart.svg').write_text(m.group(1),encoding='utf-8');sz=d.joinpath('flowchart.svg').stat().st_size;print('  [1/2] flowchart.svg written ('+str(sz)+' bytes)')"
if errorlevel 1 goto :fail

:: ── Step 2: HTML -> PDF via Edge (full font support, exact browser rendering) ─
set EDGE=
for %%E in (
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
) do if not defined EDGE (if exist %%E set EDGE=%%E)

if defined EDGE (
    echo  [2/2] Using Microsoft Edge for PDF ...

    :: Write absolute paths to a temp file to avoid single-quote conflicts in for/f
    (
        echo import pathlib
        echo p = pathlib.Path("flowchart.html").resolve()
        echo print("HTMLURL=" + p.as_uri())
        echo print("PDFPATH=" + str(p.parent / "flowchart.pdf"))
    ) > _mkpdf_paths.py
    for /f "tokens=1,* delims==" %%A in ('python _mkpdf_paths.py') do set %%A=%%B
    del _mkpdf_paths.py 2>nul

    !EDGE! --headless=new --disable-gpu --no-pdf-header-footer ^
        "--print-to-pdf=!PDFPATH!" "!HTMLURL!" >nul 2>&1

    if exist "flowchart.pdf" (
        for %%F in (flowchart.pdf) do echo         Done^^!  flowchart.pdf  ^(%%~zF bytes^)
        goto :done
    )
    echo         Edge produced no output. Falling back to cairosvg...
)

:: ── Step 2b: fallback — cairosvg (limited Unicode font support) ──────────────
echo  [2/2] Using cairosvg for PDF ...
python -c "import pathlib;d=pathlib.Path('.');__import__('cairosvg').svg2pdf(url=str(d.joinpath('flowchart.svg').resolve()),write_to=str(d.joinpath('flowchart.pdf')));sz=d.joinpath('flowchart.pdf').stat().st_size;print('  Done^^!  flowchart.pdf  ('+str(sz)+' bytes)')"
if errorlevel 1 goto :fail
goto :done

:fail
echo.
echo  FAILED — check errors above.
echo  Ensure cairosvg is installed:  pip install cairosvg
echo.
pause
exit /b 1

:done
echo.
pause
