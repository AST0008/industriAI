// app/api/process-input/route.ts
import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabaseClient';
import Papa from 'papaparse';
import { UserInput, ProcessedData, FinancialData } from '@/types';

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const contentType = request.headers.get('Content-Type') || '';

    let data: string | UserInput | UserInput[] = '';

    if (contentType.includes('application/json')) {
      data = (await request.json()) as string;
    } else if (contentType.includes('multipart/form-data')) {
      const formData = await request.formData();
      const file = formData.get('file');

      if (!file || !(file instanceof File)) {
        return NextResponse.json({ message: 'No file uploaded' }, { status: 400 });
      }

      const fileContent = await file.text();
      const parsed = Papa.parse<UserInput>(fileContent, {
        header: true,
        skipEmptyLines: true,
        transformHeader: header => header.trim(),
        transform: value => value.trim(),
      });

      if (parsed.errors.length) {
        console.error('CSV Parsing Errors:', parsed.errors);
        return NextResponse.json({ message: 'Error parsing CSV file' }, { status: 400 });
      }

      data = parsed.data;
    } else {
      return NextResponse.json({ message: 'Unsupported Media Type' }, { status: 415 });
    }

    // Send data to Flask endpoint
    const flaskEndpoint = process.env.FLASK_ENDPOINT_URL || "http://127.0.0.1:5000/process-input";

    const flaskResponse = await fetch(flaskEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!flaskResponse.ok) {
      console.error('Error processing input on Flask server');
      return NextResponse.json({ message: 'Error processing input on Flask server' }, { status: 500 });
    }

    const flaskResult: ProcessedData | ProcessedData[] = await flaskResponse.json();
    console.log('Flask result:', flaskResult);

    // Store data in Supabase
    // const { data: supabaseData, error: supabaseError } = await supabase
    // .from('financial_data')
    
    //       .insert([
    //     {
    //       user_input: data,
    //       processed_data: flaskResult,
    //       timestamp: new Date().toISOString(),
    //     },
    //   ]);

    // if (supabaseError) {
    //   console.error('Error inserting data into Supabase:', supabaseError);
    //   return NextResponse.json({ message: 'Error storing data' }, { status: 500 });
    // }

    return NextResponse.json({ message: flaskResult }, { status: 200 });
  } catch (error) {
    console.error('Error in API route:', error);
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}
