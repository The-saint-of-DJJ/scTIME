param(
    [string]$Root = "",
    [switch]$SkipHuge,
    [switch]$IncludeOptional
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
}

$logDir = Join-Path $Root "logs"
$geoDir = Join-Path $Root "data\geo"
$optionalDir = Join-Path $Root "data\optional"
$manifestPath = Join-Path $Root "CGZ_public_data_manifest.tsv"

foreach ($dir in @($Root, $logDir, $geoDir, $optionalDir)) {
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}

$entries = @(
    [pscustomobject]@{Dataset="GSE243013";Group="core";Huge=$false;ExpectedBytes=1108;Subdir="data\geo\GSE243013";FileName="GSE243013_NMF_all_group_5.csv.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243013/suppl/GSE243013_NMF_all_group_5.csv.gz";Description="NMF immune group labels"},
    [pscustomobject]@{Dataset="GSE243013";Group="core";Huge=$true;ExpectedBytes=7123039063;Subdir="data\geo\GSE243013";FileName="GSE243013_NSCLC_immune_scRNA_counts.mtx.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243013/suppl/GSE243013_NSCLC_immune_scRNA_counts.mtx.gz";Description="Processed immune scRNA count matrix"},
    [pscustomobject]@{Dataset="GSE243013";Group="core";Huge=$false;ExpectedBytes=40701210;Subdir="data\geo\GSE243013";FileName="GSE243013_NSCLC_immune_scRNA_metadata.csv.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243013/suppl/GSE243013_NSCLC_immune_scRNA_metadata.csv.gz";Description="Processed immune scRNA metadata"},
    [pscustomobject]@{Dataset="GSE243013";Group="core";Huge=$false;ExpectedBytes=15940803;Subdir="data\geo\GSE243013";FileName="GSE243013_T_with_TCR_annotation.csv.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243013/suppl/GSE243013_T_with_TCR_annotation.csv.gz";Description="T-cell TCR annotation"},
    [pscustomobject]@{Dataset="GSE243013";Group="core";Huge=$false;ExpectedBytes=21611305;Subdir="data\geo\GSE243013";FileName="GSE243013_UMAP_info.tar.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243013/suppl/GSE243013_UMAP_info.tar.gz";Description="UMAP coordinates/info"},
    [pscustomobject]@{Dataset="GSE243013";Group="core";Huge=$false;ExpectedBytes=6726478;Subdir="data\geo\GSE243013";FileName="GSE243013_barcodes.csv.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243013/suppl/GSE243013_barcodes.csv.gz";Description="scRNA barcodes"},
    [pscustomobject]@{Dataset="GSE243013";Group="core";Huge=$false;ExpectedBytes=114223;Subdir="data\geo\GSE243013";FileName="GSE243013_genes.csv.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243013/suppl/GSE243013_genes.csv.gz";Description="scRNA genes"},
    [pscustomobject]@{Dataset="GSE243013";Group="core";Huge=$false;ExpectedBytes=12907;Subdir="data\geo\GSE243013";FileName="GSE243013_series_matrix.txt.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243013/matrix/GSE243013_series_matrix.txt.gz";Description="GEO series matrix"},

    [pscustomobject]@{Dataset="GSE207422";Group="core";Huge=$false;ExpectedBytes=5629640;Subdir="data\geo\GSE207422";FileName="GSE207422_NSCLC_bulk_RNAseq_log2TPM.txt.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE207nnn/GSE207422/suppl/GSE207422_NSCLC_bulk_RNAseq_log2TPM.txt.gz";Description="Bulk RNA-seq log2TPM"},
    [pscustomobject]@{Dataset="GSE207422";Group="core";Huge=$false;ExpectedBytes=12056;Subdir="data\geo\GSE207422";FileName="GSE207422_NSCLC_bulk_RNAseq_metadata.xlsx";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE207nnn/GSE207422/suppl/GSE207422_NSCLC_bulk_RNAseq_metadata.xlsx";Description="Bulk RNA-seq metadata"},
    [pscustomobject]@{Dataset="GSE207422";Group="core";Huge=$false;ExpectedBytes=184001817;Subdir="data\geo\GSE207422";FileName="GSE207422_NSCLC_scRNAseq_UMI_matrix.txt.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE207nnn/GSE207422/suppl/GSE207422_NSCLC_scRNAseq_UMI_matrix.txt.gz";Description="scRNA UMI matrix"},
    [pscustomobject]@{Dataset="GSE207422";Group="core";Huge=$false;ExpectedBytes=11323;Subdir="data\geo\GSE207422";FileName="GSE207422_NSCLC_scRNAseq_metadata.xlsx";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE207nnn/GSE207422/suppl/GSE207422_NSCLC_scRNAseq_metadata.xlsx";Description="scRNA metadata"},
    [pscustomobject]@{Dataset="GSE207422";Group="core";Huge=$false;ExpectedBytes=4588;Subdir="data\geo\GSE207422";FileName="GSE207422_series_matrix.txt.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE207nnn/GSE207422/matrix/GSE207422_series_matrix.txt.gz";Description="GEO series matrix"},

    [pscustomobject]@{Dataset="GSE126044";Group="core";Huge=$false;ExpectedBytes=559926;Subdir="data\geo\GSE126044";FileName="GSE126044_counts.txt.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE126nnn/GSE126044/suppl/GSE126044_counts.txt.gz";Description="Bulk RNA-seq counts"},
    [pscustomobject]@{Dataset="GSE126044";Group="core";Huge=$false;ExpectedBytes=2076;Subdir="data\geo\GSE126044";FileName="GSE126044_series_matrix.txt.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE126nnn/GSE126044/matrix/GSE126044_series_matrix.txt.gz";Description="GEO series matrix"},

    [pscustomobject]@{Dataset="GSE135222";Group="core";Huge=$false;ExpectedBytes=1663896;Subdir="data\geo\GSE135222";FileName="GSE135222_GEO_RNA-seq_omicslab_exp.tsv.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE135nnn/GSE135222/suppl/GSE135222_GEO_RNA-seq_omicslab_exp.tsv.gz";Description="Bulk RNA-seq expression"},
    [pscustomobject]@{Dataset="GSE135222";Group="core";Huge=$false;ExpectedBytes=2829;Subdir="data\geo\GSE135222";FileName="GSE135222_series_matrix.txt.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE135nnn/GSE135222/matrix/GSE135222_series_matrix.txt.gz";Description="GEO series matrix"},

    [pscustomobject]@{Dataset="GSE274975";Group="core";Huge=$false;ExpectedBytes=2835943;Subdir="data\geo\GSE274975";FileName="GSE274975_raw_counts.tsv.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE274nnn/GSE274975/suppl/GSE274975_raw_counts.tsv.gz";Description="Bulk RNA-seq raw counts"},
    [pscustomobject]@{Dataset="GSE274975";Group="core";Huge=$false;ExpectedBytes=4496;Subdir="data\geo\GSE274975";FileName="GSE274975_series_matrix.txt.gz";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE274nnn/GSE274975/matrix/GSE274975_series_matrix.txt.gz";Description="GEO series matrix"},

    [pscustomobject]@{Dataset="GSE243013";Group="optional";Huge=$false;ExpectedBytes=541511680;Subdir="data\optional\GSE243013";FileName="GSE243013_RAW.tar";Url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243013/suppl/GSE243013_RAW.tar";Description="Optional GEO RAW archive"}
)

function Write-Manifest {
    param([array]$Rows)
    $Rows |
        Select-Object Dataset,Group,Huge,Status,ExpectedBytes,ActualBytes,LocalPath,Url,Description |
        Export-Csv -LiteralPath $manifestPath -Delimiter "`t" -NoTypeInformation -Encoding UTF8
}

function Get-StatusRow {
    param(
        [pscustomobject]$Entry,
        [string]$Status,
        [int64]$ActualBytes,
        [string]$LocalPath
    )
    [pscustomobject]@{
        Dataset = $Entry.Dataset
        Group = $Entry.Group
        Huge = $Entry.Huge
        Status = $Status
        ExpectedBytes = $Entry.ExpectedBytes
        ActualBytes = $ActualBytes
        LocalPath = $LocalPath
        Url = $Entry.Url
        Description = $Entry.Description
    }
}

$downloadRows = New-Object System.Collections.Generic.List[object]
$selectedEntries = $entries | Where-Object {
    ($_.Group -eq "core") -or ($IncludeOptional -and $_.Group -eq "optional")
}

foreach ($entry in $selectedEntries) {
    $targetDir = Join-Path $Root $entry.Subdir
    if (-not (Test-Path -LiteralPath $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir | Out-Null
    }

    $outFile = Join-Path $targetDir $entry.FileName
    if ($SkipHuge -and $entry.Huge) {
        $actual = if (Test-Path -LiteralPath $outFile) { (Get-Item -LiteralPath $outFile).Length } else { 0 }
        $downloadRows.Add((Get-StatusRow -Entry $entry -Status "skipped_huge" -ActualBytes $actual -LocalPath $outFile))
        Write-Manifest -Rows $downloadRows
        continue
    }

    if (Test-Path -LiteralPath $outFile) {
        $actual = (Get-Item -LiteralPath $outFile).Length
        if ($entry.ExpectedBytes -gt 0 -and $actual -eq [int64]$entry.ExpectedBytes) {
            $downloadRows.Add((Get-StatusRow -Entry $entry -Status "exists_complete" -ActualBytes $actual -LocalPath $outFile))
            Write-Manifest -Rows $downloadRows
            continue
        }
    }

    Write-Host "Downloading $($entry.Dataset) $($entry.FileName)"
    Write-Host $entry.Url

    $curlArgs = @(
        "--ssl-no-revoke",
        "--ipv4",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry", "5",
        "--retry-delay", "5",
        "--connect-timeout", "30",
        "--speed-limit", "1024",
        "--speed-time", "120",
        "--continue-at", "-",
        "--output", $outFile,
        $entry.Url
    )

    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed with exit code $LASTEXITCODE for $($entry.Url)"
    }

    $actualAfter = (Get-Item -LiteralPath $outFile).Length
    $statusAfter = if ($entry.ExpectedBytes -gt 0 -and $actualAfter -ne [int64]$entry.ExpectedBytes) {
        "downloaded_size_mismatch"
    } else {
        "downloaded_complete"
    }

    $downloadRows.Add((Get-StatusRow -Entry $entry -Status $statusAfter -ActualBytes $actualAfter -LocalPath $outFile))
    Write-Manifest -Rows $downloadRows
}

Write-Manifest -Rows $downloadRows
Write-Host "Manifest written to $manifestPath"
