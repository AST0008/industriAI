// app/portfolio/page.tsx

"use client";

import React, { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";

// Define the Holding interface
interface Holding {
  company: string;
  weight: number; // Percentage
  esgContribution: number;
  riskContribution: number;
  returnContribution: number;
}

// Define the PortfolioData interface
interface PortfolioData {
  basicComposition: {
    totalInvested: number;
    numberOfHoldings: number;
    averagePositionSize: number;
  };
  esgProfile: {
    portfolioEsgScore: number;
    weightedCarbonFootprint: number;
    greenCompaniesAllocation: number;
  };
  riskAndReturnMetrics: {
    expectedPortfolioRoi: number;
    portfolioRiskScore: number;
    riskAdjustedReturn: number;
  };
  diversificationMetrics: {
    diversificationScore: number;
    largestPosition: number;
    smallestPosition: number;
  };
  holdingsBreakdown: Holding[];
}

const PortfolioPage: React.FC = () => {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(
    null
  );
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Hardcoded portfolio data
  const hardcodedData: PortfolioData = {
    basicComposition: {
      totalInvested: 1000000.0,
      numberOfHoldings: 2,
      averagePositionSize: 500000.0,
    },
    esgProfile: {
      portfolioEsgScore: 0.5,
      weightedCarbonFootprint: 30.5,
      greenCompaniesAllocation: 60.0,
    },
    riskAndReturnMetrics: {
      expectedPortfolioRoi: -3.0,
      portfolioRiskScore: 0.4,
      riskAdjustedReturn: -7.5,
    },
    diversificationMetrics: {
      diversificationScore: 0.8,
      largestPosition: 50.0,
      smallestPosition: 50.0,
    },
    holdingsBreakdown: [
      {
        company: "Company_A",
        weight: 50.0, // Percentage
        esgContribution: 0.3,
        riskContribution: 0.2,
        returnContribution: -3.5,
      },
      {
        company: "Company_B",
        weight: 50.0, // Percentage
        esgContribution: 0.2,
        riskContribution: 0.3,
        returnContribution: -2.5,
      },
    ],
  };

  // Simulate data fetching
  useEffect(() => {
    const fetchPortfolioData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Simulate network delay
        await new Promise((resolve) => setTimeout(resolve, 1000));

        // Set the hardcoded data
        setPortfolioData(hardcodedData);
      } catch (err: any) {
        setError(err.message || "An unknown error occurred.");
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolioData();
  }, []);

  // Handle optimization action
  const handleOptimize = () => {
    // For demonstration, we'll alert and reset the data
    alert("Optimize Portfolio button clicked!");
    // Optionally, you can reset or update the portfolio data here
    // setPortfolioData(null);
  };

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-8 text-center text-gray-800">
        Sustainability Dashboard
      </h1>

      {/* Loading State */}
      {loading && (
        <p className="text-center text-gray-600">Loading portfolio data...</p>
      )}

      {/* Error State */}
      {error && <p className="text-center text-red-600">{error}</p>}

      {/* Portfolio Data */}
      {portfolioData && (
        <Card className="p-6 shadow-lg mb-6">
          <h2 className="text-2xl font-semibold mb-4 text-gray-700">
            Portfolio Analysis Report
          </h2>

          {/* 1. Basic Portfolio Composition */}
          <section className="mb-6">
            <h3 className="text-xl font-semibold mb-2">
              1. Basic Portfolio Composition
            </h3>
            <ul className="list-disc list-inside space-y-1 text-gray-600">
              <li>
                <strong>Total Invested:</strong> $
                {portfolioData.basicComposition.totalInvested.toLocaleString()}
              </li>
              <li>
                <strong>Number of Holdings:</strong>{" "}
                {portfolioData.basicComposition.numberOfHoldings}
              </li>
              <li>
                <strong>Average Position Size:</strong> $
                {portfolioData.basicComposition.averagePositionSize.toLocaleString()}
              </li>
            </ul>
          </section>

          {/* 2. ESG Profile */}
          <section className="mb-6">
            <h3 className="text-xl font-semibold mb-2">2. ESG Profile</h3>
            <ul className="list-disc list-inside space-y-1 text-gray-600">
              <li>
                <strong>Portfolio ESG Score:</strong>{" "}
                {portfolioData.esgProfile.portfolioEsgScore.toFixed(3)}
              </li>
              <li>
                <strong>Weighted Carbon Footprint:</strong>{" "}
                {portfolioData.esgProfile.weightedCarbonFootprint.toFixed(2)}
              </li>
              <li>
                <strong>Green Companies Allocation:</strong>{" "}
                {portfolioData.esgProfile.greenCompaniesAllocation.toFixed(1)}%
              </li>
            </ul>
          </section>

          {/* 3. Risk and Return Metrics */}
          <section className="mb-6">
            <h3 className="text-xl font-semibold mb-2">
              3. Risk and Return Metrics
            </h3>
            <ul className="list-disc list-inside space-y-1 text-gray-600">
              <li>
                <strong>Expected Portfolio ROI:</strong>{" "}
                {portfolioData.riskAndReturnMetrics.expectedPortfolioRoi.toFixed(
                  2
                )}
                %
              </li>
              <li>
                <strong>Portfolio Risk Score:</strong>{" "}
                {portfolioData.riskAndReturnMetrics.portfolioRiskScore.toFixed(
                  3
                )}
              </li>
              <li>
                <strong>Risk-Adjusted Return:</strong>{" "}
                {portfolioData.riskAndReturnMetrics.riskAdjustedReturn.toFixed(
                  3
                )}
              </li>
            </ul>
          </section>

          {/* 4. Diversification Metrics */}
          <section className="mb-6">
            <h3 className="text-xl font-semibold mb-2">
              4. Diversification Metrics
            </h3>
            <ul className="list-disc list-inside space-y-1 text-gray-600">
              <li>
                <strong>Diversification Score:</strong>{" "}
                {portfolioData.diversificationMetrics.diversificationScore.toFixed(
                  3
                )}
              </li>
              <li>
                <strong>Largest Position:</strong>{" "}
                {portfolioData.diversificationMetrics.largestPosition.toFixed(
                  1
                )}
                %
              </li>
              <li>
                <strong>Smallest Position:</strong>{" "}
                {portfolioData.diversificationMetrics.smallestPosition.toFixed(
                  1
                )}
                %
              </li>
            </ul>
          </section>

          {/* 5. Holdings Breakdown */}
          <section>
            <h3 className="text-xl font-semibold mb-2">
              5. Holdings Breakdown
            </h3>
            {portfolioData.holdingsBreakdown.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Company</TableHead>
                    <TableHead>Weight (%)</TableHead>
                    <TableHead>ESG Contribution</TableHead>
                    <TableHead>Risk Contribution</TableHead>
                    <TableHead>Return Contribution</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {portfolioData.holdingsBreakdown.map((holding, index) => (
                    <TableRow key={index}>
                      <TableCell>{holding.company}</TableCell>
                      <TableCell>{holding.weight.toFixed(1)}</TableCell>
                      <TableCell>
                        {holding.esgContribution.toFixed(3)}
                      </TableCell>
                      <TableCell>
                        {holding.riskContribution.toFixed(3)}
                      </TableCell>
                      <TableCell>
                        {holding.returnContribution.toFixed(3)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <p className="text-gray-600">No holdings available.</p>
            )}
          </section>
        </Card>
      )}
    </div>
  );
};

export default PortfolioPage;
