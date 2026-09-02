export type Dataset = {
    id: string;
    title: string;
    fileName: string;
    format: "JSONL" | "CSV" | "Alpaca" | "ShareGPT";
    category: string;
    fileSize: number; // in MB or KB
    totalRows: number;
    validationStatus: "Valid" | "Invalid" | "Error";
    sftStatus: "Ready" | "Training" | "Queued" | "Failed";
    uploadedAt: string; // ISO 8601 date string
    uploadedBy: string;
}

export type TrainingRun = {
    runId: string;
    targetModel: string;
    targetModelBase: string;
    datasetVersion: string;
    status: "COMPLETED" | "RUNNING" | "PENDING" | "FAILED";
    trainLoss: number | null;
    evalLoss: number | null;
    created: Date;
}