// app/dashboard/page.tsx

"use client";

import { Card } from "@/components/ui/card";
import { Bar, Pie, Radar, Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Filler,
} from "chart.js";
import { supabase } from "@/lib/supabaseClient";
import { FinancialData, UserInput, ProcessedData } from "@/types";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Filler
);

// Helper function to convert grades to numerical values

const DashboardPage: React.FC = () => {
  const [financialData, setFinancialData] = useState<FinancialData | null>(
    null
  );
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const fetchFinancialData = async () => {
      try {
        const { data, error } = await supabase
          .from("financial_data")
          .select("*")
          .order("timestamp", { ascending: false })
          .limit(1);

        if (error) {
          console.error("Error fetching data from Supabase:", error);
          setError("Failed to load data.");
        } else {
          setFinancialData(data[0] || null);
          console.log(financialData);
        }
      } catch (err) {
        console.error("Unexpected error:", err);
        setError("An unexpected error occurred.");
      } finally {
        setLoading(false);
      }
    };

    fetchFinancialData();
  }, [financialData]);

  if (loading) {
    return (
      <div className="p-6 bg-gray-100 min-h-screen flex items-center justify-center">
        <div className="text-center text-gray-600">
          Loading dashboard data...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-gray-100 min-h-screen flex items-center justify-center">
        <div className="text-center text-red-600">{error}</div>
      </div>
    );
  }

  if (!financialData) {
    return (
      <div className="p-6 bg-gray-100 min-h-screen">
        <h1 className="text-3xl font-bold mb-8 text-center text-gray-800">
          Sustainability Dashboard
        </h1>
        <div className="text-center text-gray-600">
          No data available. Please input your financial data.
        </div>
      </div>
    );
  }

  const { user_input, processed_data } = financialData;

  // Handle both single and multiple user inputs
  const inputs: UserInput[] = Array.isArray(user_input)
    ? user_input
    : [user_input];
  const processedResults: ProcessedData[] = Array.isArray(processed_data)
    ? processed_data
    : [processed_data];

  // For simplicity, using the first entry
  const latestInput = inputs[0];
  console.log("Latest Input:", latestInput);

  const latestProcessed = processedResults[0];

  // Prepare chart data based on latestInput
  const barData = {
    labels: ["Net Assets Rate (M USD)"],
    datasets: [
      {
        label: "Net Assets Rate",
        data: [latestInput.netAssetsRate],
        backgroundColor: ["#4CAF50"],
      },
    ],
  };

  const pieData = {
    labels: [
      "Fossil Fuel Grade",
      "Deforestation Grade",
      "Civilian Firearms Grade",
      "Military Weapons Grade",
      "Tobacco Grade",
      "Prison Grade",
    ],
    datasets: [
      {
        label: "Grades",
        data: [
          latestInput.fossilFuel,
          latestInput.deforestation,
          latestInput.civilianFirearms,
          latestInput.militaryWeapons,
          latestInput.tobacco,
          latestInput.prison,
        ],
        backgroundColor: [
          "#4CAF50",
          "#FF9800",
          "#FFC107",
          "#F44336",
          "#9C27B0",
          "#00BCD4",
        ],
      },
    ],
  };

  const radarData = {
    labels: [
      "Fossil Fuel",
      "Deforestation",
      "Civilian Firearms",
      "Military Weapons",
      "Tobacco",
      "Prison",
    ],
    datasets: [
      {
        label: "Grades",
        data: [
          latestInput.fossilFuel,
          latestInput.deforestation,
          latestInput.civilianFirearms,
          latestInput.militaryWeapons,
          latestInput.tobacco,
          latestInput.prison,
        ],
        backgroundColor: "rgba(33, 150, 243, 0.2)",
        borderColor: "#2196F3",
        borderWidth: 1,
        pointBackgroundColor: "#2196F3",
        fill: true,
      },
    ],
  };

  const doughnutData = {
    labels: ["Carbon Footprint", "Remaining"],
    datasets: [
      {
        data: [
          Number(latestInput.relativeCarbonFootprint),
          100 - Number(latestInput.relativeCarbonFootprint),
        ],
        backgroundColor: ["#FF6384", "#E0E0E0"],
        hoverBackgroundColor: ["#FF6384", "#E0E0E0"],
        borderWidth: 0,
      },
    ],
  };

  const doughnutOptions = {
    rotation: -90, // Start angle
    circumference: 180, // Sweep angle
    cutout: "70%", // Thickness of the doughnut
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: false,
      },
      title: {
        display: true,
        text: "Relative Carbon Footprint",
        position: "bottom" as const,
        font: {
          size: 16,
        },
      },
    },
    responsive: true,
    maintainAspectRatio: false,
  };

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-8 text-center text-gray-800">
        Sustainability Dashboard
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Net Assets Rate Bar Chart */}
        <Card className="p-6 shadow-lg">
          <h2 className="text-xl font-semibold mb-4 text-gray-700">
            Net Assets Rate
          </h2>
          <Bar
            data={barData}
            options={{
              indexAxis: "y" as const,
              responsive: true,
              plugins: {
                legend: {
                  display: false,
                },
                title: {
                  display: true,
                  text: "Net Assets Rate Overview",
                },
              },
              scales: {
                x: {
                  beginAtZero: true,
                },
              },
            }}
          />
        </Card>

        {/* Relative Carbon Footprint Doughnut Chart */}
        <Card className="p-6 shadow-lg flex flex-col items-center justify-center">
          <h2 className="text-xl font-semibold mb-4 text-gray-700">
            Relative Carbon Footprint
          </h2>
          <div className="w-full h-48 relative">
            <Doughnut data={doughnutData} options={doughnutOptions} />
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-2xl font-bold text-gray-800">
                {latestInput.relativeCarbonFootprint}%
              </span>
            </div>
          </div>
        </Card>

        {/* Environmental Grades Pie Chart */}
        <Card className="p-6 shadow-lg">
          <h2 className="text-xl font-semibold mb-4 text-gray-700">
            Environmental Grades
          </h2>
          <Pie
            data={pieData}
            options={{
              responsive: true,
              plugins: {
                legend: {
                  position: "right" as const,
                },
                title: {
                  display: true,
                  text: "Grades Distribution",
                },
              },
            }}
          />
        </Card>

        {/* Grades Overview Radar Chart */}
        <Card className="p-6 shadow-lg lg:col-span-1">
          <h2 className="text-xl font-semibold mb-4 text-gray-700">
            Grades Overview
          </h2>
          <div className="w-full h-64">
            <Radar
              data={radarData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  r: {
                    suggestedMin: 0,
                    suggestedMax: 5,
                    ticks: {
                      stepSize: 1,
                    },
                    angleLines: {
                      display: true,
                    },
                    grid: {
                      color: "#e0e0e0",
                    },
                    pointLabels: {
                      font: {
                        size: 12,
                      },
                    },
                  },
                },
                plugins: {
                  legend: {
                    display: false,
                  },
                  title: {
                    display: false,
                    text: "Grades by Category",
                  },
                },
              }}
            />
          </div>
        </Card>
      </div>

      {/* Additional Insights Section */}
      <Card className="p-6 mt-8 shadow-lg">
        <h2 className="text-xl font-semibold mb-4 text-gray-700">
          Additional Insights
        </h2>
        <ul className="list-disc list-inside space-y-2 text-gray-600">
          <li>
            <strong>Net Assets Rate:</strong> $
            {latestInput.netAssetsRate.toLocaleString()}M
          </li>
          <li>
            <strong>Relative Carbon Footprint:</strong>{" "}
            {latestInput.relativeCarbonFootprint}%
          </li>
          <li>
            <strong>Grades:</strong>
            <ul className="list-none pl-4">
              <li>
                <span className="capitalize">Fossil Fuel:</span>{" "}
                <span className="font-semibold">{latestInput.fossilFuel}</span>
              </li>
              <li>
                <span className="capitalize">Deforestation:</span>{" "}
                <span className="font-semibold">
                  {latestInput.deforestation}
                </span>
              </li>
              <li>
                <span className="capitalize">Civilian Firearms:</span>{" "}
                <span className="font-semibold">
                  {latestInput.civilianFirearms}
                </span>
              </li>
              <li>
                <span className="capitalize">Military Weapons:</span>{" "}
                <span className="font-semibold">
                  {latestInput.militaryWeapons}
                </span>
              </li>
              <li>
                <span className="capitalize">Tobacco:</span>{" "}
                <span className="font-semibold">{latestInput.tobacco}</span>
              </li>
              <li>
                <span className="capitalize">Prison:</span>{" "}
                <span className="font-semibold">{latestInput.prison}</span>
              </li>
            </ul>
          </li>
        </ul>
      </Card>

      {/* Action Button */}
      <div className="flex justify-center mt-8">
        <button
          onClick={() => router.push("/optimize")}
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg shadow-lg transition-all"
        >
          Optimize Sustainability
        </button>
      </div>
    </div>
  );
};

export default DashboardPage;
