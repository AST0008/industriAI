"use client";

import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

const metrics = [
    { label: "Shareclass Name", value: "Name of the shareclass." },
    { label: "Ticker", value: "Ticker of the shareclass." },
    { label: "Fund Name", value: "Name of the mutual fund or ETF." },
    { label: "Asset Manager", value: "The asset manager that offers this shareclass." },
    { label: "Shareclass Type", value: "Type of shareclass, either open-end fund or ETF." },
    { label: "Shareclass Inception Date", value: "The date of inception of this shareclass." },
    { label: "Category Group", value: "The category group this fund is assigned to, based on its portfolio investment profile." },
    { label: "Sustainability Mandate", value: "'Y' if this fund has a sustainability mandate." },
    { label: "US-SIF Member", value: "'Y' if this fund is a member of The Forum for Sustainable and Responsible Investing, also known as US-SIF." },
    { label: "Oldest Shareclass Inception Date", value: "The oldest date of inception for any shareclass for this fund." },
    { label: "Shareclass Tickers", value: "The tickers of any other open-end or exchange-traded shareclasses offered for this fund." },
    { label: "Portfolio Holdings As-Of Date", value: "The date the portfolio holdings were reported by the fund manager." },
    { label: "Fund Net Assets", value: "The amount of fund assets, as calculated using the market value of the holdings of the portfolio, as of the portfolio holdings date." },
    { label: "Percent Rated", value: "The percent of fund assets rated by Invest Your Values. Our methodology is restricted to long-position equities, so 'percent rated' is approximate to the percent of fund assets invested directly in stocks." },
    { label: "Fossil Fuel Grade", value: "The fossil fuel grade for this fund, based on total fossil fuel exposure." },
    { label: "Fossil Fuel Holdings, Count", value: "The number of holdings in this fund's portfolio found on the five fossil fuel screen lists." },
    { label: "Fossil Fuel Holdings, Weight", value: "The percent of fund assets invested in holdings found on the five fossil fuel screen lists." },
    { label: "Fossil Fuel Holdings, Asset", value: "The amount in USD of fund assets invested in holdings found on the five fossil fuel screen lists." },
    { label: "Carbon Underground 200, Count", value: "The number of holdings in this fund's portfolio found on the Carbon Underground 200 screen list." },
    { label: "Carbon Underground 200, Weight", value: "The percent of fund assets invested in holdings found on the Carbon Underground 200 screen list." },
    { label: "Carbon Underground 200, Asset", value: "The amount in USD of fund assets invested in holdings found on the Carbon Underground 200 screen list." },
    { label: "Coal Industry, Count", value: "The number of holdings in this fund's portfolio found on the coal industry screen list." },
    { label: "Coal Industry, Weight", value: "The percent of fund assets invested in holdings found on the coal industry screen list." },
    { label: "Coal Industry, Asset", value: "The amount in USD of fund assets invested in holdings found on the coal industry screen list." },
    { label: "Oil/Gas Industry, Count", value: "The number of holdings in this fund's portfolio found on the oil/gas industry screen list." },
    { label: "Oil/Gas Industry, Weight", value: "The percent of fund assets invested in holdings found on the oil/gas industry screen list." },
    { label: "Oil/Gas Industry, Asset", value: "The amount in USD of fund assets invested in holdings found on the oil/gas industry screen list." },
    { label: "Macroclimate 30 Coal-Fired Utilities, Count", value: "The number of holdings in this fund's portfolio found on the Macroclimate 30 coal-fired utilities screen list." },
    { label: "Macroclimate 30 Coal-Fired Utilities, Weight", value: "The percent of fund assets invested in holdings found on the Macroclimate 30 coal-fired utilities screen list." },
    { label: "Macroclimate 30 Coal-Fired Utilities, Asset", value: "The amount in USD of fund assets invested in holdings found on the Macroclimate 30 coal-fired utilities screen list." },
    { label: "Relative Carbon Footprint (tonnes CO2 / $1M USD invested)", value: "Expresses the greenhouse gas footprint of an investment sum." },
    { label: "Relative Carbon Intensity (tonnes CO2 / $1M USD revenue)", value: "Expresses the carbon efficiency of a portfolio." },
    { label: "Total Financed Emissions Scope 1+2 (tCO2e)", value: "Measures the absolute greenhouse gas footprint of a portfolio in tons of carbon dioxide equivalents." },
    { label: "Total Financed Emissions Scope 1+2+3 (tCO2e)", value: "Includes direct and indirect emissions." },
    { label: "Gender Equality Grade", value: "The gender equality grade for this fund, based on the Equileap gender equality scores of the companies in the portfolio." },
    { label: "Gender Equality Score (out of 100 points)", value: "Calculated by averaging the Equileap gender equality company scores of the portfolio holdings." },
    { label: "Deforestation Grade", value: "The deforestation grade for this fund, based on exposure to three deforestation-risk screen lists." },
    { label: "Tobacco Grade", value: "The tobacco grade for this fund, based on exposure to two tobacco screen lists." },
    { label: "Weapon Free Grade", value: "The military weapons grade for this fund, based on exposure to three military weapon screen lists." }
  ];

export default function AdditionalMetricsPage() {
  return (
    <div className="p-6 bg-background h-screen flex flex-col">
      <Card className="mb-4">
        <div className="p-4">
          <h1 className="text-2xl font-semibold">Additional Metrics</h1>
          <p className="text-sm text-muted-foreground">
            Here are detailed metrics for your financial data.
          </p>
        </div>
      </Card>

      <ScrollArea className="flex-grow">
        <Card className="p-4">
          <div className="space-y-4">
            {metrics.map((metric, index) => (
              <div key={index} className="border-b pb-4">
                <h2 className="text-lg font-medium">{metric.label}</h2>
                <p className="text-sm text-muted-foreground">{metric.value}</p>
              </div>
            ))}
          </div>
        </Card>
      </ScrollArea>
    </div>
  );
}
