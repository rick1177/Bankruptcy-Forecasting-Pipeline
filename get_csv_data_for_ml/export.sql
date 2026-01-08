COPY (
    WITH
        Get_Base AS (
            SELECT
                "year", inn, Ogrn, Region, Region_Taxcode, Creation_Date, Dissolution_Date, Age, Eligible,
                Exemption_Criteria, Financial, Filed, Imputed, Simplified, Articulated, Totals_Adjustment,
                Outlier, Okved, Okved_Section, Okpo, Okopf, Okogu, Okfc, Oktmo,

                COALESCE(Line_1100, CASE WHEN Line_1600 IS NOT NULL THEN 0 END) AS Line_1100,
                COALESCE(Line_1200, CASE WHEN Line_1600 IS NOT NULL THEN 0 END) AS Line_1200,
                COALESCE(Line_1300, CASE WHEN Line_1600 IS NOT NULL THEN 0 END) AS Line_1300,
                COALESCE(Line_1400, CASE WHEN Line_1600 IS NOT NULL THEN 0 END) AS Line_1400,
                COALESCE(Line_1500, CASE WHEN Line_1600 IS NOT NULL THEN 0 END) AS Line_1500,
                Line_1600 AS Line_1600,
                Line_1700 AS Line_1700,

                COALESCE(Line_2100, CASE WHEN Line_2400 IS NOT NULL THEN 0 END) AS Line_2100,
                COALESCE(Line_2200, CASE WHEN Line_2400 IS NOT NULL THEN 0 END) AS Line_2200,
                COALESCE(Line_2300, CASE WHEN Line_2400 IS NOT NULL THEN 0 END) AS Line_2300,
                Line_2400 AS Line_2400,

                (((COALESCE(Line_2310, 0) + COALESCE(Line_2320, 0)) + COALESCE(Line_2340, 0))) AS All_Other_Revenue,

                CAST(Line_1200 AS DOUBLE) / NULLIF(Line_1600, 0) AS K_1200_1600,
                CAST(Line_1300 AS DOUBLE) / NULLIF(Line_1600, 0) AS K_1300_1600,
                CAST(Line_1500 AS DOUBLE) / NULLIF(Line_1600, 0) AS K_1500_1600,
                CAST((COALESCE(Line_1400, 0) + COALESCE(Line_1500, 0)) AS DOUBLE) / NULLIF(Line_1600, 0) AS K_14001500_1600,
                Line_1300 < 0 AS K_Neg_1300,
                COALESCE(CAST(Line_2400 AS DOUBLE) / NULLIF(Line_1600, 0), 0) AS K_2400_1600,
                COALESCE(CAST(Line_2400 AS DOUBLE) / NULLIF(Line_2100, 0), 0) AS K_2400_2100,
                COALESCE(CAST(Line_2200 AS DOUBLE) / NULLIF(Line_2100, 0), 0) AS K_2200_2100,
                COALESCE(CAST(Line_2300 AS DOUBLE) / NULLIF(Line_2100, 0), 0) AS K_2300_2100

            FROM rfsd_all
            WHERE year > 2018
        ),

        get_active_from_base AS (
            SELECT
                *,
                CASE
                    WHEN all_other_revenue = 0 AND line_2100 = 0 THEN FALSE
                    WHEN EXISTS (
                        SELECT 1
                        FROM unnest(ARRAY[
                            line_1100, line_1200, line_1300, line_1400, line_1500, line_1600, line_1700,
                            line_2100, line_2200, line_2300, line_2400
                        ]) AS x(v)
                        WHERE v IS NULL
                    ) AND year > EXTRACT(YEAR FROM CAST(creation_date AS DATE)) THEN FALSE
                    WHEN NOT EXISTS (
                        SELECT 1
                        FROM unnest(ARRAY[
                            line_1100, line_1200, line_1300, line_1400, line_1500, line_1600, line_1700,
                            line_2100, line_2200, line_2300, line_2400
                        ]) AS x(v)
                        WHERE v IS NULL
                    ) AND (line_2100 <> 0 OR all_other_revenue <> 0) THEN TRUE
                    WHEN year = EXTRACT(YEAR FROM CAST(creation_date AS DATE)) THEN TRUE
                    ELSE FALSE
                END AS active
            FROM get_base
        ),

        exceptions AS (
            SELECT DISTINCT
                CAST(inn AS TEXT) AS inn,
                CAST(year AS INT) AS year
            FROM hidden_reporting
        ),

        candidates AS (
            SELECT
                a1.inn,
                a1.year + 1 AS base_year,

                a1.active AS active_t1,
                a1.line_1100 AS line_1100_t1,
                a1.line_1200 AS line_1200_t1,
                a1.line_1300 AS line_1300_t1,
                a1.line_1400 AS line_1400_t1,
                a1.line_1500 AS line_1500_t1,
                a1.line_1600 AS line_1600_t1,
                a1.line_1700 AS line_1700_t1,
                a1.line_2100 AS line_2100_t1,
                a1.line_2200 AS line_2200_t1,
                a1.line_2300 AS line_2300_t1,
                a1.line_2400 AS line_2400_t1,

                a1.k_1200_1600 AS k_1200_1600_t1,
                a1.k_1300_1600 AS k_1300_1600_t1,
                a1.k_1500_1600 AS k_1500_1600_t1,
                a1.k_14001500_1600 AS k_14001500_1600_t1,
                a1.k_neg_1300 AS k_neg_1300_t1,
                a1.k_2400_1600 AS k_2400_1600_t1,
                a1.k_2400_2100 AS k_2400_2100_t1,
                a1.k_2200_2100 AS k_2200_2100_t1,
                a1.k_2300_2100 AS k_2300_2100_t1,

                a2.active AS active_t2,
                a2.line_1100 AS line_1100_t2,
                a2.line_1200 AS line_1200_t2,
                a2.line_1300 AS line_1300_t2,
                a2.line_1400 AS line_1400_t2,
                a2.line_1500 AS line_1500_t2,
                a2.line_1600 AS line_1600_t2,
                a2.line_1700 AS line_1700_t2,
                a2.line_2100 AS line_2100_t2,
                a2.line_2200 AS line_2200_t2,
                a2.line_2300 AS line_2300_t2,
                a2.line_2400 AS line_2400_t2,

                a2.k_1200_1600 AS k_1200_1600_t2,
                a2.k_1300_1600 AS k_1300_1600_t2,
                a2.k_1500_1600 AS k_1500_1600_t2,
                a2.k_14001500_1600 AS k_14001500_1600_t2,
                a2.k_neg_1300 AS k_neg_1300_t2,
                a2.k_2400_1600 AS k_2400_1600_t2,
                a2.k_2400_2100 AS k_2400_2100_t2,
                a2.k_2200_2100 AS k_2200_2100_t2,
                a2.k_2300_2100 AS k_2300_2100_t2,

                a3.active AS active_t3,
                a3.line_1100 AS line_1100_t3,
                a3.line_1200 AS line_1200_t3,
                a3.line_1300 AS line_1300_t3,
                a3.line_1400 AS line_1400_t3,
                a3.line_1500 AS line_1500_t3,
                a3.line_1600 AS line_1600_t3,
                a3.line_1700 AS line_1700_t3,
                a3.line_2100 AS line_2100_t3,
                a3.line_2200 AS line_2200_t3,
                a3.line_2300 AS line_2300_t3,
                a3.line_2400 AS line_2400_t3,

                a3.k_1200_1600 AS k_1200_1600_t3,
                a3.k_1300_1600 AS k_1300_1600_t3,
                a3.k_1500_1600 AS k_1500_1600_t3,
                a3.k_14001500_1600 AS k_14001500_1600_t3,
                a3.k_neg_1300 AS k_neg_1300_t3,
                a3.k_2400_1600 AS k_2400_1600_t3,
                a3.k_2400_2100 AS k_2400_2100_t3,
                a3.k_2200_2100 AS k_2200_2100_t3,
                a3.k_2300_2100 AS k_2300_2100_t3

            FROM get_active_from_base a1
            JOIN get_active_from_base a2 ON a2.inn = a1.inn AND a2.year = a1.year - 1
            JOIN get_active_from_base a3 ON a3.inn = a1.inn AND a3.year = a1.year - 2
            WHERE a1.active = TRUE AND a2.active = TRUE AND a3.active = TRUE
        ),

        with_t AS (
            SELECT
                c.*,
                t.active AS active_t,
                CASE
                    WHEN e.inn IS NOT NULL THEN 0
                    WHEN t.inn IS NOT NULL THEN CASE WHEN t.active = FALSE THEN 1 ELSE 0 END
                    ELSE 1
                END AS bankrupt_t,
                CASE
                    WHEN c.base_year BETWEEN 2021 AND 2023 THEN 'train'
                    WHEN c.base_year = 2024 THEN 'test'
                    ELSE 'other'
                END AS dataset_split
            FROM candidates c
            LEFT JOIN get_active_from_base t ON t.inn = c.inn AND t.year = c.base_year
            LEFT JOIN exceptions e ON e.inn = c.inn AND e.year = c.base_year
        ),

        final AS (
            SELECT
                *,
                (k_1300_1600_t1 - k_1300_1600_t2) AS d_k_1300_1600_t1,
                (k_2400_1600_t1 - k_2400_1600_t2) AS d_k_2400_1600_t1,
                (k_2400_2100_t1 - k_2400_2100_t2) AS d_k_2400_2100_t1,
                (k_2400_1600_t1 + k_2400_1600_t2 + k_2400_1600_t3) / 3.0 AS k_2400_1600_trend_3y,
                (k_2400_2100_t1 + k_2400_2100_t2 + k_2400_2100_t3) / 3.0 AS k_2400_2100_trend_3y,
                (line_2400_t1 < line_2400_t2 AND line_2400_t2 < line_2400_t3) AS net_profit_decline_2y_pre,
                (line_2100_t1 < line_2100_t2 AND line_2100_t2 < line_2100_t3) AS revenue_decline_2y_pre,
                (line_1300_t1 < line_1300_t2 AND line_1300_t2 < line_1300_t3) AS equity_decline_2y_pre
            FROM with_t
        )

    SELECT *
    FROM final
    WHERE dataset_split IN ('train','test')
) TO '__OUT_PATH__'
  (FORMAT CSV, HEADER TRUE, DELIMITER ',', NULL '', QUOTE '"', ESCAPE '"');
