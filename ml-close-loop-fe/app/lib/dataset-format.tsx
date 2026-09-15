import * as XLSX from "xlsx";

export type DetectedFormat = "JSONL" | "CSV" | "Excel (.xlsx)" | "Unknown Format";

export function detectFormat(file: File): DetectedFormat {
	const ext = file.name.split(".").pop()?.toLowerCase();
	switch (ext) {
		case "jsonl":
			return "JSONL";
		case "csv":
			return "CSV";
		case "xlsx":
			return "Excel (.xlsx)";
		default:
			return "Unknown Format";
	}
}

function countJsonlLines(text: string): number {
	return text.split(/\r?\n/).filter((line) => line.trim().length > 0).length;
}

async function csvToRecords(file: File): Promise<Record<string, unknown>[]> {
	const text = await file.text();
	const rows = parseCsvRows(text).filter((row) => row.some((cell) => cell.trim().length > 0));
	if (rows.length === 0) return [];

	const headers = rows[0];
	const records: Record<string, unknown>[] = [];

	for (let i = 1; i < rows.length; i++) {
		const row = rows[i];
		const record: Record<string, unknown> = {};
		for (let j = 0; j < headers.length; j++) {
			record[headers[j] ?? `column_${j + 1}`] = row[j] ?? "";
		}
		records.push(record);
	}

	return records;
}

function parseCsvRows(text: string): string[][] {
	const rows: string[][] = [];
	let row: string[] = [];
	let cell = "";
	let inQuotes = false;

	for (let i = 0; i < text.length; i++) {
		const char = text[i];
		const nextChar = text[i + 1];

		if (char === '"') {
			if (inQuotes && nextChar === '"') {
				cell += '"';
				i++;
			} else {
				inQuotes = !inQuotes;
			}
			continue;
		}

		if (!inQuotes && char === ",") {
			row.push(cell);
			cell = "";
			continue;
		}

		if (!inQuotes && (char === "\n" || char === "\r")) {
			if (char === "\r" && nextChar === "\n") i++;
			row.push(cell);
			rows.push(row);
			row = [];
			cell = "";
			continue;
		}

		cell += char;
	}

	if (cell.length > 0 || row.length > 0) {
		row.push(cell);
		rows.push(row);
	}

	return rows;
}

async function excelToRecords(file: File): Promise<Record<string, unknown>[]> {
	const buffer = await file.arrayBuffer();
	const workbook = XLSX.read(buffer, { type: "array" });
	const sheet = workbook.Sheets[workbook.SheetNames[0]];
	return XLSX.utils.sheet_to_json(sheet, { defval: null });
}

function recordsToJsonl(records: Record<string, unknown>[]): string {
	return records.map((r) => JSON.stringify(r)).join("\n");
}

/**
 * Deteksi format file, konversi ke JSONL kalau perlu, lalu hitung jumlah sample.
 * JSONL dihitung per baris non-kosong; CSV/Excel dihitung per baris data (setelah konversi).
 */
export async function processDatasetFile(file: File): Promise<{
	format: DetectedFormat;
	jsonlText: string;
	sampleCount: number;
}> {
	const format = detectFormat(file);

	if (format === "JSONL") {
		const text = await file.text();
		return { format, jsonlText: text, sampleCount: countJsonlLines(text) };
	}

	if (format === "CSV") {
		const records = await csvToRecords(file);
		return { format, jsonlText: recordsToJsonl(records), sampleCount: records.length };
	}

	if (format === "Excel (.xlsx)") {
		const records = await excelToRecords(file);
		return { format, jsonlText: recordsToJsonl(records), sampleCount: records.length };
	}

	throw new Error("Format file tidak didukung. Gunakan JSONL, CSV, atau XLSX.");
}