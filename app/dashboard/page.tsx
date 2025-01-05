"use client";

import { Card } from "@/components/ui/card";
import { Bar, Pie, Line } from "react-chartjs-2";
import { useRouter } from "next/navigation";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  PointElement,
  LineElement,
  ArcElement,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  PointElement,
  LineElement,
  ArcElement
);

const mockData = {
  annualRevenue: 5000000,
  operatingExpenses: 2000000,
  grossProfitMargin: 0.6,
  netIncome: 1200000,
  totalAssets: 8000000,
  totalLiabilities: 3000000,
  cashFlow: 1500000,
  capitalExpenditure: 700000,
  revenueTrend: [4000000, 4500000, 5000000, 5500000, 6000000],
  expensesTrend: [1500000, 1800000, 2000000, 2300000, 2500000],
};

export default function DetailedDashboard() {
  const router = useRouter();

  const handleOptimizeClick = () => {
    router.push("/optimize");
  };

  const barData = {
    labels: [
      "Annual Revenue",
      "Operating Expenses",
      "Net Income",
      "Total Assets",
      "Total Liabilities",
      "Cash Flow",
      "Capital Expenditure",
    ],
    datasets: [
      {
        label: "Financial Metrics (USD)",
        data: [
          mockData.annualRevenue,
          mockData.operatingExpenses,
          mockData.netIncome,
          mockData.totalAssets,
          mockData.totalLiabilities,
          mockData.cashFlow,
          mockData.capitalExpenditure,
        ],
        backgroundColor: [
          "#4CAF50",
          "#2196F3",
          "#FFC107",
          "#FF5722",
          "#9C27B0",
          "#00BCD4",
          "#E91E63",
        ],
      },
    ],
  };

  const pieData = {
    labels: ["Gross Profit Margin", "Operating Expenses"],
    datasets: [
      {
        label: "Profit Distribution",
        data: [
          mockData.grossProfitMargin * mockData.annualRevenue,
          mockData.operatingExpenses,
        ],
        backgroundColor: ["#4CAF50", "#FF5722"],
      },
    ],
  };

  const lineData = {
    labels: ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"],
    datasets: [
      {
        label: "Revenue Trend",
        data: mockData.revenueTrend,
        borderColor: "#4CAF50",
        backgroundColor: "rgba(76, 175, 80, 0.5)",
        tension: 0.4,
      },
      {
        label: "Expenses Trend",
        data: mockData.expensesTrend,
        borderColor: "#FF5722",
        backgroundColor: "rgba(255, 87, 34, 0.5)",
        tension: 0.4,
      },
    ],
  };

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-8 text-center text-gray-800">
        Detailed Financial Dashboard
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Financial Metrics Chart */}
        <Card className="p-6 lg:col-span-2 shadow-lg">
          <h2 className="text-xl font-semibold mb-4 text-gray-700">
            Financial Metrics Overview
          </h2>
          <Bar data={barData} options={{ responsive: true }} />
        </Card>

        {/* Pie Chart */}
        <Card className="p-6 shadow-lg">
          <h2 className="text-xl font-semibold mb-4 text-gray-700">
            Profit Distribution
          </h2>
          <Pie data={pieData} options={{ responsive: true }} />
        </Card>

        {/* Revenue vs. Expenses Trend */}
        <Card className="p-6 lg:col-span-3 shadow-lg">
          <h2 className="text-xl font-semibold mb-4 text-gray-700">
            Revenue vs. Expenses Trend
          </h2>
          <Line data={lineData} options={{ responsive: true }} />
        </Card>
      </div>

      {/* Key Insights Section */}
      <Card className="p-6 mt-8 shadow-lg">
        <h2 className="text-xl font-semibold mb-4 text-gray-700">Key Insights</h2>
        <ul className="list-disc list-inside space-y-2 text-gray-600">
          <li>
            <strong>Annual Revenue:</strong> ${mockData.annualRevenue.toLocaleString()}
          </li>
          <li>
            <strong>Operating Expenses:</strong> ${mockData.operatingExpenses.toLocaleString()}
          </li>
          <li>
            <strong>Net Income:</strong> ${mockData.netIncome.toLocaleString()}
          </li>
          <li>
            <strong>Total Assets:</strong> ${mockData.totalAssets.toLocaleString()}
          </li>
          <li>
            <strong>Total Liabilities:</strong> ${mockData.totalLiabilities.toLocaleString()}
          </li>
          <li>
            <strong>Cash Flow from Operations:</strong> ${mockData.cashFlow.toLocaleString()}
          </li>
          <li>
            <strong>Capital Expenditure:</strong> ${mockData.capitalExpenditure.toLocaleString()}
          </li>
        </ul>
      </Card>

      {/* Optimize Button */}
      <div className="flex justify-center mt-8">
        <button
          onClick={handleOptimizeClick}
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg shadow-lg transition-all"
        >
          Optimize
        </button>
      </div>
    </div>
  );
}
