-- Use the COVID_portfolio database
USE COVID_portfolio;

/* -- Checking the imported data in both 'deaths' and 'vaccinations' tables 
-- to ensure everything has been loaded correctly and looks in order. */

SELECT 
    *
FROM
    deaths
ORDER BY location , date;

-- Check the first 50 records in the vaccinations table, ordered by location and date
SELECT 
    *
FROM
    vaccinations
ORDER BY location , date
LIMIT 50;

-- Select relevant columns to get an overview of the first 50 records in the deaths table
SELECT 
    location,
    date,
    total_cases,
    new_cases,
    total_deaths,
    population
FROM
    deaths
ORDER BY location , date
LIMIT 50;

-- Analyze total cases vs total deaths to calculate the likelihood of dying (death rate)
-- from COVID-19 and what percentage of the population got infected (infection rate).
SELECT 
    location,
    date,
    total_cases,
    total_deaths,
    100 * (total_deaths / total_cases) AS death_rate,
    population,
    100 * (total_cases / population) AS infected_pop_percent
FROM
    deaths
WHERE
    location LIKE '%france%'
ORDER BY location , date;

-- Identify countries with the highest infection rate relative to their population
-- and calculate death rates across different continents.
SELECT 
    location,
    continent,
    MAX(population) AS pop,
    MAX(CAST(total_cases AS UNSIGNED)) AS infection_count,
    MAX(total_deaths) AS total_deaths,
    MAX(100 * (total_deaths / total_cases)) AS death_percent,
    MAX(100 * (total_cases / population)) AS infected_pop_percent
FROM
    deaths
WHERE
    continent IS NOT NULL
GROUP BY location , continent
ORDER BY continent DESC;-- Order by continent in descending order

SELECT 
    location,
    MAX(population) AS pop,
    MAX(CAST(total_cases AS UNSIGNED)) AS infection_count,
    MAX(total_deaths) AS total_deaths,
    MAX(100 * (total_deaths / total_cases)) AS death_percent,
    MAX(100 * (total_cases / population)) AS infected_pop_percent
FROM
    deaths
WHERE
    continent IS NULL
GROUP BY location
ORDER BY total_deaths DESC;-- Order by total deaths in descending order

SELECT 
    date,
    SUM(new_cases) AS new_cases,
    SUM(new_deaths) AS new_deaths,
    100 * (SUM(new_deaths) / SUM(new_cases)) AS death_percent
FROM
    deaths
WHERE
    continent IS NOT NULL
GROUP BY date
HAVING new_cases IS NOT NULL
ORDER BY date;  -- Order by date

-- Explore vaccination data by analyzing the vaccinated proportion of the population
-- and calculate the cumulative sum of vaccinations over time for each location.
SELECT d.continent, d.location, d.date, d.population, v.new_vaccinations,
SUM(v.new_vaccinations) OVER (PARTITION BY d.location ORDER BY d.location, d.date) AS cum_vaccinations  -- Cumulative sum of vaccinations per location
FROM deaths AS d
JOIN vaccinations AS v  -- Join deaths and vaccinations tables by location and date
ON d.location = v.location AND d.date = v.date
WHERE d.continent IS NOT NULL  -- Only include valid continents
ORDER BY d.location, d.date;  -- Order by location and date

-- Use a common table expression (CTE) to calculate cumulative vaccinations
-- and the percentage of the population vaccinated for each location.
WITH vaccinated_people (continent, location, date, population, new_vaccinations, cum_vaccinations) AS (
    SELECT d.continent, d.location, d.date, d.population, v.new_vaccinations,
    SUM(v.new_vaccinations) OVER (PARTITION BY d.location ORDER BY d.location, d.date) AS cum_vaccinations  -- Cumulative vaccinations
    FROM deaths AS d
    JOIN vaccinations AS v  -- Join deaths and vaccinations tables by location and date
    ON d.location = v.location AND d.date = v.date
    WHERE d.continent IS NOT NULL  -- Only include valid continents
    ORDER BY d.location, d.date
)
SELECT *, 100 * (cum_vaccinations / population) AS cum_vaccinated_percent  -- Calculate vaccinated percentage
FROM vaccinated_people;

-- Create a view to store vaccination data for later use in visualizations.
CREATE VIEW PercentPeopleVaccinated AS
SELECT d.continent, d.location, d.date, d.population, v.new_vaccinations,
SUM(v.new_vaccinations) OVER (PARTITION BY d.location ORDER BY d.location, d.date) AS cum_vaccinations  -- Cumulative sum of vaccinations
FROM deaths AS d
JOIN vaccinations AS v
ON d.location = v.location AND d.date = v.date
WHERE d.continent IS NOT NULL;

-- Query the view to check the results.
SELECT 
    *
FROM
    COVID_portfolio.percentpeoplevaccinated;
