// types/index.ts
export interface UserInput {
  netAssetsRate: number ;
  fossilFuel: number;
  deforestation: number;
  relativeCarbonFootprint: number | string; // 1-100
  civilianFirearms: number;
  militaryWeapons: number;
  tobacco: number;
  prison: number;
}

export interface ProcessedData {
  message: string;
  // Add more fields as needed based on Flask response
}

export interface FinancialData {
  id: string;
  user_input: UserInput | UserInput[];
  processed_data: ProcessedData | ProcessedData[];
  timestamp: string;
}
