CREATE OR REPLACE VIEW hidden_reporting AS
SELECT DISTINCT inn, "year"
	FROM
		read_csv_auto(
		'D:/Code/new_bfo/get_db_vsh/dds_bfo_bal_fin_protect.csv');

CREATE
OR REPLACE VIEW rfsd_all AS
SELECT *
	FROM
		read_parquet([
  				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2011\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2012\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2013\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2014\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2015\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2016\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2017\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2018\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2019\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2020\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2021\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2022\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2023\\part-0.parquet',
				'D:\\Code\\new_bfo\\data_cache\\datasets--irlspbru--RFSD\\snapshots\\0211c9905c8f2bbf61029ebe7fe5bea4f2c6184a\\RFSD\\year=2024\\part-0.parquet'
					]);
                    

-- ============================================================
-- Обновляем запрос с подстройкой под структуру модели обучения
-- ============================================================
-- ------------------------------------------------------------
-- ВИТРИНА ДЛЯ ML (банкротство): из rfsd_all
--
-- Твоя постановка таргета:
-- Компания считается «банкротом» в году t, если:
--   (A) в данных есть строка за год t и active_t = FALSE
--   ИЛИ
--   (B) компания работала в t-3,t-2,t-1 (active=TRUE),
--     а строки за t ВООБЩЕ НЕТ (исчезла из данных)
--
-- Исключения (whitelist):
-- Если строка за t отсутствует, но (inn,t) есть в списке исключений,
-- то НЕ считаем банкротом (bankrupt_t = 0).
-- Ты подаёшь список: (inn, year).
--
-- 1 строка = (inn, base_year=t)
-- Признаки: ТОЛЬКО за t-1,t-2,t-3
-- Train/Test:
-- TRAIN: base_year in [2021..2023] (нужна история с 2018)
-- TEST : base_year = 2024
-- ============================================================
-- Для модели
WITH
	Get_Base AS (
		SELECT
			"year", inn, Ogrn, Region, Region_Taxcode, Creation_Date, Dissolution_Date, Age, Eligible,
			Exemption_Criteria, Financial, Filed, Imputed, Simplified, Articulated, Totals_Adjustment,
			Outlier, Okved, Okved_Section, Okpo, Okopf, Okogu, Okfc, Oktmo,

-- Баланс (B)
			COALESCE(Line_1100,
					 CASE
						 WHEN Line_1600 IS NOT NULL
							 THEN 0
					 END) AS Line_1100,                       --B_noncurrent_assets
			COALESCE(Line_1200,
					 CASE
						 WHEN Line_1600 IS NOT NULL
							 THEN 0
					 END) AS Line_1200,                       --B_current_assets
			COALESCE(Line_1300,
					 CASE
						 WHEN Line_1600 IS NOT NULL
							 THEN 0
					 END) AS Line_1300,                       --B_total_equity
			COALESCE(Line_1400,
					 CASE
						 WHEN Line_1600 IS NOT NULL
							 THEN 0
					 END) AS Line_1400,                       --B_longterm_liab
			COALESCE(Line_1500,
					 CASE
						 WHEN Line_1600 IS NOT NULL
							 THEN 0
					 END) AS Line_1500,                       --B_shortterm_liab
			Line_1600 AS Line_1600,                           --B_assets
			Line_1700 AS Line_1700,                           --B_liab

-- Финрез (PL)
			COALESCE(Line_2100,
					 CASE
						 WHEN Line_2400 IS NOT NULL
							 THEN 0
					 END) AS Line_2100,                       --PL_revenue
			COALESCE(Line_2200,
					 CASE
						 WHEN Line_2400 IS NOT NULL
							 THEN 0
					 END) AS Line_2200,                       --PL_profit_sales
			COALESCE(Line_2300,
					 CASE
						 WHEN Line_2400 IS NOT NULL
							 THEN 0
					 END) AS Line_2300,                       --PL_before_tax
			Line_2400 AS Line_2400,                           --PL_net_profit

			(
				(
					COALESCE(Line_2310, 0) +
					COALESCE(Line_2320, 0)) +
				COALESCE(Line_2340, 0)
				) AS All_Other_Revenue,                       --PL_all_othe_revenue

-- Расчётные коэффициенты
			CAST(Line_1200 AS double precision) /
			NULLIF(Line_1600, 0) AS K_1200_1600,              --K_current_assets_share
			CAST(
					Line_1300 AS double precision) / NULLIF(
					Line_1600, 0) AS K_1300_1600,             --K_equity_share
			CAST(
					Line_1500 AS double precision) / NULLIF(
					Line_1600, 0) AS K_1500_1600,             --K_short_liabilities_share
			CAST(
					(
						COALESCE(
								Line_1400, 0) + COALESCE(
								Line_1500, 0)) AS double precision) / NULLIF(
					Line_1600, 0) AS K_14001500_1600,         --K_total_liabilities_share
			Line_1300 < 0 AS K_Neg_1300,                      --K_negative_equity
			COALESCE(
					CAST(
							Line_2400 AS double precision) / NULLIF(
							Line_1600, 0), 0) AS K_2400_1600, --K_return_on_assets
			COALESCE(
					CAST(
							Line_2400 AS double precision) / NULLIF(
							Line_2100, 0), 0) AS K_2400_2100, --K_net_profit_margin
			COALESCE(
					CAST(
							Line_2200 AS double precision) / NULLIF(
							Line_2100, 0), 0) AS K_2200_2100, --K_net_profit_sales
			COALESCE(
					CAST(
							Line_2300 AS double precision) / NULLIF(
							Line_2100, 0), 0) AS K_2300_2100  --K_profit_before_tax_margin

			FROM
				Rfsd_All
			WHERE YEAR > 2018),

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
					WHERE v IS NULL) AND
					YEAR > EXTRACT(YEAR FROM CAST(creation_date AS DATE)) THEN FALSE
				WHEN NOT EXISTS (
					SELECT 1
					FROM unnest(ARRAY[
						line_1100, line_1200, line_1300, line_1400, line_1500, line_1600, line_1700,
						line_2100, line_2200, line_2300, line_2400
						]) AS x(v)
					WHERE v IS NULL) AND
					(line_2100 <> 0 OR all_other_revenue <> 0) THEN TRUE
				WHEN YEAR = EXTRACT(YEAR FROM CAST(creation_date AS DATE)) THEN TRUE
				ELSE FALSE
			END AS active
		FROM get_base),

-- ============================================================
-- Список исключений (whitelist) — ЗАПОЛНИШЬ СЮДА
-- Формат: (inn, year)
-- Пример:
-- ('7701234567', 2024),
-- ('7800000000', 2022)
-- ============================================================
	exceptions AS (
		SELECT DISTINCT
			CAST(inn AS TEXT) AS inn,
			CAST(year AS INT) AS year
		FROM hidden_reporting
	),


-- ============================================================
-- Формируем кандидатов событий по году t:
-- Берём год (t-1) как опорный и строим base_year = (t-1)+1
-- Требуем, чтобы t-1,t-2,t-3 существуют и active=TRUE
-- ============================================================
candidates AS (
    SELECT
        a1.inn,
        a1.year + 1 AS base_year,

        -- t-1
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

        -- t-2
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

        -- t-3
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
    JOIN get_active_from_base a2
      ON a2.inn = a1.inn AND a2.year = a1.year - 1
    JOIN get_active_from_base a3
      ON a3.inn = a1.inn AND a3.year = a1.year - 2

    WHERE
      a1.active = TRUE
      AND a2.active = TRUE
      AND a3.active = TRUE
),

-- Подтягиваем год t, если он есть (может отсутствовать)
with_t AS (
	SELECT
		c.*,
		t.active AS active_t,


	-- ТАРГЕТ (bankrupt_t) с OVERRIDE-логикой исключений:
	-- (0) если (inn,t) есть в exceptions -> всегда НЕ банкрот (0)
	-- (A) иначе, если год t есть -> банкротство = (active_t = FALSE)
	-- (B) иначе (год t отсутствует) -> банкротство = 1
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
	LEFT JOIN get_active_from_base t
		ON t.inn = c.inn AND t.year = c.base_year
	LEFT JOIN exceptions e
		ON e.inn = c.inn AND e.year = c.base_year
),

final AS (
    SELECT
        *,

        -- динамика (t-1) - (t-2)
        (k_1300_1600_t1 - k_1300_1600_t2) AS d_k_1300_1600_t1,
        (k_2400_1600_t1 - k_2400_1600_t2) AS d_k_2400_1600_t1,
        (k_2400_2100_t1 - k_2400_2100_t2) AS d_k_2400_2100_t1,

        -- тренды 3 года (t-1,t-2,t-3)
        (k_2400_1600_t1 + k_2400_1600_t2 + k_2400_1600_t3) / 3.0 AS k_2400_1600_trend_3y,
        (k_2400_2100_t1 + k_2400_2100_t2 + k_2400_2100_t3) / 3.0 AS k_2400_2100_trend_3y,

        -- флаги ухудшения (2 года подряд до события): t-3 → t-2 → t-1
        (line_2400_t1 < line_2400_t2 AND line_2400_t2 < line_2400_t3) AS net_profit_decline_2y_pre,
        (line_2100_t1 < line_2100_t2 AND line_2100_t2 < line_2100_t3) AS revenue_decline_2y_pre,
        (line_1300_t1 < line_1300_t2 AND line_1300_t2 < line_1300_t3) AS equity_decline_2y_pre

    FROM with_t
)

SELECT *
FROM final
WHERE
    dataset_split IN ('train','test');

-- ============================================================
-- ЧТО ТЫ МНЕ СЮДА ДАЁШЬ ДАЛЬШЕ
-- 1) Список исключений (whitelist) (inn, year) — вставишь в CTE exceptions.
-- 2) Если хочешь учитывать «исчезновение» только при eligible_t1=1 или filed_t1=1 — добавим фильтр.
-- ============================================================
