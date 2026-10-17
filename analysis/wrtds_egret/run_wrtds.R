## ─────────────────────────────────────────────────────────────────
## WRTDS Flow-Normalization for MRC Stations (v3)
## ─────────────────────────────────────────────────────────────────
## Usage:
##   Rscript run_wrtds.R              # fast mode: estimation + tables
##   Rscript run_wrtds.R --bootstrap  # full mode: adds bootstrap CIs
##   Rscript run_wrtds.R --station TanChau           # single station fast
##   Rscript run_wrtds.R --station TanChau --bootstrap # single station + bootstrap
##   Rscript run_wrtds.R --recache    # force re-estimation (ignores saved RDS)
## ─────────────────────────────────────────────────────────────────

library(EGRET)
library(EGRETci)

# ── Parse command-line args ──
all_args   <- commandArgs(trailingOnly = FALSE)
cli_args   <- commandArgs(trailingOnly = TRUE)
RUN_BOOT   <- "--bootstrap" %in% cli_args
RECACHE    <- "--recache"   %in% cli_args
ONLY_STN   <- NULL
if ("--station" %in% cli_args) {
  idx <- which(cli_args == "--station")
  if (idx < length(cli_args)) ONLY_STN <- cli_args[idx + 1]
}

file_arg <- grep("^--file=", all_args, value = TRUE)
if (length(file_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE)
  data_dir <- dirname(script_path)
} else if (dir.exists("analysis/wrtds_egret")) {
  data_dir <- normalizePath("analysis/wrtds_egret", mustWork = FALSE)
} else {
  data_dir <- getwd()
}
out_dir  <- file.path(data_dir, "outputs")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

BREAK_YEAR <- 2004

cat(sprintf("Mode: %s | Station filter: %s | Cache: %s\n",
            ifelse(RUN_BOOT, "FULL (bootstrap)", "FAST (no bootstrap)"),
            ifelse(is.null(ONLY_STN), "ALL", ONLY_STN),
            ifelse(RECACHE, "RECACHE (force re-estimation)", "USE CACHE")))

# ── Helper: build EGRET Daily dataframe from our CSV ──
build_daily <- function(csv_path, q_unit_factor = 1) {
  raw <- read.csv(csv_path, stringsAsFactors = FALSE)
  raw$Date <- as.Date(raw$dateTime, format = "%Y-%m-%d")
  raw <- raw[!is.na(raw$Date) & !is.na(raw$Q), ]
  raw <- raw[order(raw$Date), ]
  raw$Q <- raw$Q * q_unit_factor

  Daily <- data.frame(
    Date      = raw$Date,
    Q         = raw$Q,
    LogQ      = log(raw$Q),
    Julian    = as.numeric(format(raw$Date, "%j")),
    Month     = as.numeric(format(raw$Date, "%m")),
    Day       = as.numeric(format(raw$Date, "%d")),
    DecYear   = decimalDate(raw$Date),
    MonthSeq  = ((as.numeric(format(raw$Date, "%Y")) - 1850) * 12) +
                 as.numeric(format(raw$Date, "%m")),
    Qualifier = rep("", nrow(raw)),
    i         = seq_len(nrow(raw)),
    Q7        = as.numeric(stats::filter(raw$Q, rep(1/7, 7),   sides = 1)),
    Q30       = as.numeric(stats::filter(raw$Q, rep(1/30, 30), sides = 1)),
    stringsAsFactors = FALSE
  )
  Daily$Q7[is.na(Daily$Q7)]   <- Daily$Q[is.na(Daily$Q7)]
  Daily$Q30[is.na(Daily$Q30)] <- Daily$Q[is.na(Daily$Q30)]
  return(Daily)
}

# ── Helper: build EGRET Sample dataframe from our CSV ──
build_sample <- function(csv_path) {
  raw <- read.csv(csv_path, stringsAsFactors = FALSE)
  raw$Date <- as.Date(raw$dateTime, format = "%Y-%m-%d")
  raw <- raw[!is.na(raw$Date) & !is.na(raw$ConcLow), ]
  raw <- raw[raw$ConcLow > 0, ]
  raw <- raw[order(raw$Date), ]

  Sample <- data.frame(
    Date      = raw$Date,
    ConcLow   = raw$ConcLow,
    ConcHigh  = raw$ConcHigh,
    Uncen     = raw$Uncen,
    ConcAve   = (raw$ConcLow + raw$ConcHigh) / 2,
    Julian    = as.numeric(format(raw$Date, "%j")),
    Month     = as.numeric(format(raw$Date, "%m")),
    Day       = as.numeric(format(raw$Date, "%d")),
    DecYear   = decimalDate(raw$Date),
    MonthSeq  = ((as.numeric(format(raw$Date, "%Y")) - 1850) * 12) +
                 as.numeric(format(raw$Date, "%m")),
    SinDY     = sin(2 * pi * decimalDate(raw$Date)),
    CosDY     = cos(2 * pi * decimalDate(raw$Date)),
    stringsAsFactors = FALSE
  )
  return(Sample)
}

# ── Station configs ──
stations <- list(
  list(id = "TanChau",  name = "Tan Chau",  area = 545000),
  list(id = "MyThuan",  name = "My Thuan",  area = 370000),
  list(id = "CanTho",   name = "Can Tho",   area = 340000),
  list(id = "MyTho",    name = "My Tho",    area = 360000)
)

params <- list(
  list(code = "COD", label = "COD (KMnO4)", units = "mg/L"),
  list(code = "NO3", label = "NO3-N",       units = "mg/L")
)

summary_rows <- list()

for (stn in stations) {
  # Station filter
  if (!is.null(ONLY_STN) && stn$id != ONLY_STN) next

  for (par in params) {
    t0 <- Sys.time()
    cat("\n======================================================\n")
    cat(sprintf("  %s - %s\n", stn$name, par$label))
    cat("======================================================\n")

    daily_file  <- file.path(data_dir, paste0(stn$id, "_Daily.csv"))
    sample_file <- file.path(data_dir, paste0(stn$id, "_", par$code, "_Sample.csv"))

    if (!file.exists(daily_file) || !file.exists(sample_file)) {
      cat("  WARNING: Missing input file, skipping.\n")
      next
    }

    # ── Build INFO ──
    INFO <- data.frame(
      siteNumber     = stn$id,
      shortName      = paste0(stn$name, " - Mekong Delta"),
      paramShortName = par$label,
      param.units    = par$units,
      drainSqKm      = stn$area,
      staAbbrev      = stn$id,
      constitAbbrev  = par$code,
      paStart        = 10,
      paLong         = 12,
      stringsAsFactors = FALSE
    )

    Daily  <- build_daily(daily_file)
    Sample <- build_sample(sample_file)
    cat(sprintf("  Daily: %d days | Sample: %d obs\n", nrow(Daily), nrow(Sample)))

    # ── Merge into eList (populates Q/LogQ in Sample rows) ──
    eList <- mergeReport(INFO, Daily, Sample, verbose = FALSE)

    # ── Model estimation (with RDS caching) ──
    rds_path <- file.path(out_dir, paste0(stn$id, "_", par$code, "_eList.rds"))
    if (!RECACHE && file.exists(rds_path)) {
      cat("  Loading cached eList from RDS...\n")
      eList <- readRDS(rds_path)
    } else {
      cat("  Running WRTDS model estimation...\n")
      eList <- modelEstimation(eList, verbose = FALSE)
      cat("  Model estimation complete.\n")

      # ── WRTDSKalman ──
      eList <- tryCatch(WRTDSKalman(eList),
                        error = function(e) { cat(sprintf("  Note: WRTDSKalman skipped (%s)\n", e$message)); eList })

      saveRDS(eList, rds_path)
      cat(sprintf("  eList cached -> %s\n", basename(rds_path)))
    }

    # ── runGroups: Phase comparison ──
    # windowSide=0 for stationary flow normalization
    # flowBreak=TRUE because upstream dam regulation changed discharge distribution
    cat("  Running runGroups (wall + flowBreak at 2004)...\n")
    groupResults <- tryCatch({
      runGroups(eList,
                windowSide = 0,
                group1firstYear = 1986, group1lastYear = 2004,
                group2firstYear = 2005, group2lastYear = 2022,
                wall = TRUE,
                sample1EndDate = paste0(BREAK_YEAR, "-12-31"),
                flowBreak = TRUE,
                Q1EndDate = paste0(BREAK_YEAR, "-12-31"),
                paStart = 10, paLong = 12)
    }, error = function(e) {
      cat(sprintf("  Note: runGroups failed (%s)\n", e$message))
      NULL
    })

    if (!is.null(groupResults)) {
      cat("  runGroups results:\n")
      print(groupResults)
      write.csv(as.data.frame(t(unlist(groupResults))),
                file.path(out_dir, paste0(stn$id, "_", par$code, "_GroupResults.csv")),
                row.names = FALSE)
    }

    # ── runPairs: endpoint comparison ──
    cat("  Running runPairs (wall + flowBreak at 2004)...\n")
    pairResults <- tryCatch({
      runPairs(eList,
               year1 = 1990, year2 = 2020,
               windowSide = 0,
               wall = TRUE,
               sample1EndDate = paste0(BREAK_YEAR, "-12-31"),
               flowBreak = TRUE,
               Q1EndDate = paste0(BREAK_YEAR, "-12-31"),
               paStart = 10, paLong = 12)
    }, error = function(e) {
      cat(sprintf("  Note: runPairs failed (%s)\n", e$message))
      NULL
    })

    if (!is.null(pairResults)) {
      cat("  runPairs results:\n")
      print(pairResults)
      write.csv(as.data.frame(t(unlist(pairResults))),
                file.path(out_dir, paste0(stn$id, "_", par$code, "_PairResults.csv")),
                row.names = FALSE)
    }

    # ── Bootstrap (only in --bootstrap mode) ──
    if (RUN_BOOT) {
      # Preferred method: runPairsBoot (handles non-stationary discharge)
      if (!is.null(pairResults)) {
        cat("  Running runPairsBoot (nBoot=100)...\n")
        tryCatch({
          bootPair <- runPairsBoot(eList, pairResults, nBoot = 100,
                                   blockLength = 200)
          # Save full bootstrap output
          boot_csv <- file.path(out_dir,
                                paste0(stn$id, "_", par$code, "_PairsBoot.csv"))
          write.csv(data.frame(
            Station   = stn$name,
            Parameter = par$code,
            xConc     = bootPair$xConc,
            xFlux     = bootPair$xFlux,
            pConc     = bootPair$pConc,
            pFlux     = bootPair$pFlux
          ), boot_csv, row.names = FALSE)
          cat(sprintf("  PairsBoot: median %%change Conc=%.2f%%, Flux=%.2f%%\n",
                      median(bootPair$pConc, na.rm = TRUE),
                      median(bootPair$pFlux, na.rm = TRUE)))
          cat(sprintf("  PairsBoot: 90%% CI Conc=[%.3f, %.3f], Flux=[%.1f, %.1f]\n",
                      quantile(bootPair$xConc, 0.05, na.rm = TRUE),
                      quantile(bootPair$xConc, 0.95, na.rm = TRUE),
                      quantile(bootPair$xFlux, 0.05, na.rm = TRUE),
                      quantile(bootPair$xFlux, 0.95, na.rm = TRUE)))
        }, error = function(e) {
          cat(sprintf("  Note: runPairsBoot failed (%s)\n", e$message))
        })
      }

      # Also run WBT for compatibility (writes self-contained text report)
      cat("  Running WBT (nBoot=100)...\n")
      tryCatch({
        caseSetUp <- trendSetUp(eList,
                                year1 = BREAK_YEAR + 1,
                                year2 = 2022,
                                nBoot = 100,
                                bootBreak = 80,
                                blockLength = 200)
        eBoot <- wBT(eList, caseSetUp,
                      fileName = file.path(out_dir,
                                           paste0(stn$id, "_", par$code, "_WBT.txt")))
        # eBoot$pConc/pFlux are VECTORS of % change per replicate (not p-values)
        # eBoot$bootOut is the structured results data frame
        # eBoot$wordsOut is the human-readable summary
        cat("  WBT summary:\n")
        cat(eBoot$wordsOut, sep = "\n")

        write.csv(eBoot$bootOut,
                  file.path(out_dir, paste0(stn$id, "_", par$code, "_BootOut.csv")),
                  row.names = FALSE)
      }, error = function(e) {
        cat(sprintf("  Note: WBT skipped (%s)\n", e$message))
      })
    }

    # ── Annual results table ──
    results <- tableResults(eList, fluxUnit = 6)

    if (!is.null(results) && nrow(results) > 0) {
      col_names <- colnames(results)
      fn_conc_idx <- grep("FN.*[Cc]onc|FNConc", col_names)
      year_idx    <- grep("[Yy]ear", col_names)

      if (length(fn_conc_idx) > 0 && length(year_idx) > 0) {
        fn_col_name   <- col_names[fn_conc_idx[1]]
        year_col_name <- col_names[year_idx[1]]
        pre  <- results[results[[year_col_name]] <= BREAK_YEAR, fn_col_name]
        post <- results[results[[year_col_name]] >  BREAK_YEAR, fn_col_name]
        pre_mean  <- mean(pre, na.rm = TRUE)
        post_mean <- mean(post, na.rm = TRUE)
        pct_change <- (post_mean - pre_mean) / pre_mean * 100

        cat(sprintf("  FN Conc <=2004: %.3f  |  >2004: %.3f  |  Change: %+.1f%%\n",
                    pre_mean, post_mean, pct_change))

        summary_rows[[length(summary_rows) + 1]] <- data.frame(
          Station    = stn$name,
          Parameter  = par$code,
          FN_Pre2004 = round(pre_mean, 4),
          FN_Post2004 = round(post_mean, 4),
          Pct_Change = round(pct_change, 1),
          stringsAsFactors = FALSE
        )
      }
    }

    write.csv(results, file.path(out_dir, paste0(stn$id, "_", par$code, "_AnnualResults.csv")),
              row.names = FALSE)

    elapsed <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1)
    cat(sprintf("  [%s_%s complete in %.1f min]\n", stn$id, par$code, elapsed))
  }
}

# ── Summary ──
if (length(summary_rows) > 0) {
  summary_df <- do.call(rbind, summary_rows)
  write.csv(summary_df, file.path(out_dir, "WRTDS_FlowNormalized_Summary.csv"),
            row.names = FALSE)
  cat("\n\n=== WRTDS Flow-Normalized Summary ===\n")
  print(summary_df)
}

cat("\nDone.\n")
