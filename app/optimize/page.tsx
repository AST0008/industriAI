"use client";

import { Card } from "@/components/ui/card";

export default function OptimizeResults() {
  const optimizationResults = {
    investments: [
      { company: "A", amount: 40000, decision: 1 },
      { company: "B", amount: 0, decision: 0 },
      { company: "C", amount: 30000, decision: 1 },
      { company: "D", amount: 30000, decision: 1 },
    ],
    totalROI: 15400,
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-4xl font-bold mb-8 text-center text-gray-800">
        Optimization Results
      </h1>

      {/* Results Summary */}
      <Card className="p-6 mb-8 shadow-md hover:shadow-lg transition-shadow duration-300">
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">Summary</h2>
        <p className="text-lg text-gray-600">
          <strong className="text-gray-800">Total ROI:</strong>{" "}
          <span className="text-green-600 font-bold">
            ${optimizationResults.totalROI.toLocaleString()}
          </span>
        </p>
      </Card>

      {/* Investment Table */}
      <Card className="p-6 shadow-md hover:shadow-lg transition-shadow duration-300">
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">
          Investment Breakdown
        </h2>
        {/* Table for larger screens */}
        <div className="hidden md:block">
          <table className="min-w-full bg-white border border-gray-200 rounded-lg">
            <thead>
              <tr className="bg-gray-100">
                <th className="px-6 py-3 text-left font-medium text-gray-700 border-b">
                  Company
                </th>
                <th className="px-6 py-3 text-right font-medium text-gray-700 border-b">
                  Amount Invested
                </th>
                <th className="px-6 py-3 text-center font-medium text-gray-700 border-b">
                  Decision
                </th>
              </tr>
            </thead>
            <tbody>
              {optimizationResults.investments.map((investment, index) => (
                <tr
                  key={index}
                  className="border-b hover:bg-gray-50 transition-colors duration-200"
                >
                  <td className="px-6 py-4">{investment.company}</td>
                  <td className="px-6 py-4 text-right">
                    ${investment.amount.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 text-center">
                    {investment.decision ? (
                      <span className="text-green-600 font-bold">Yes</span>
                    ) : (
                      <span className="text-red-600 font-bold">No</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Card layout for mobile screens */}
        <div className="block md:hidden space-y-4">
          {optimizationResults.investments.map((investment, index) => (
            <div
              key={index}
              className="p-4 bg-white border border-gray-200 rounded-lg shadow-md"
            >
              <p className="text-lg font-medium text-gray-800">
                <strong>Company:</strong> {investment.company}
              </p>
              <p className="text-lg text-gray-600">
                <strong>Amount Invested:</strong>{" "}
                <span className="text-gray-800">
                  ${investment.amount.toLocaleString()}
                </span>
              </p>
              <p className="text-lg">
                <strong>Decision:</strong>{" "}
                {investment.decision ? (
                  <span className="text-green-600 font-bold">Yes</span>
                ) : (
                  <span className="text-red-600 font-bold">No</span>
                )}
              </p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
