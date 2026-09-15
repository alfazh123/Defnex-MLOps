import { FieldGroup } from "../../ui/field";
import { CheckCircle2 } from "lucide-react";
import { Separator } from "../../ui/separator";
import { Button } from "~/components/ui/button";
import { ScrollArea } from "~/components/ui/scroll-area";

export type StepFiveValueProps = {
    datasetId: string;
    datasetVersion: string;
    sourceMethod: "file" | "hf";
    sourceFile: File | null;
    hfRepoId: string;
    targetSchemaId: string;
    detectedSample: number;
}

export function DatasetStepFive({value, setStepActive, submitDataset}: {
    value: StepFiveValueProps;
    setStepActive: (step: number) => void;
    submitDataset: () => void;
}) {

    return (
        <>
            <ScrollArea className="max-h-125">
                <FieldGroup>
                    <div className="flex flex-col gap-4 border p-4 rounded-xl bg-white shadow-sm">
                        <div className="flex justify-between items-start">
                            <div className="flex flex-col gap-1">
                                <h3 className="font-bold text-sm tracking-wide text-gray-500">
                                    IMMUTABLE VERSION INTAKE MANIFEST
                                </h3>
                                <p className="text-xs text-muted-foreground">
                                    Confirm configuration prior to cryptographically locking this dataset version
                                </p>
                            </div>
                            <div className="flex items-center gap-1 bg-green-50 text-green-700 border border-green-200 px-2.5 py-1 rounded-full text-xs font-semibold">
                                <CheckCircle2 className="h-3.5 w-3.5" />
                                Ready
                            </div>
                        </div>

                        <Separator />

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                            <div className="flex flex-col gap-1">
                                <span className="text-xs text-gray-900 font-bold tracking-wider">
                                    DATASET ID
                                </span>
                                <span className="font-bold text-gray-500">
                                    {value.datasetId || "--"}
                                </span>
                            </div>
                            <div className="flex flex-col gap-1">
                                <span className="text-xs text-gray-900 font-bold tracking-wider">
                                    VERSION TAG
                                </span>
                                <span className="font-bold text-gray-700">
                                    {value.datasetVersion || "--"}
                                </span>
                            </div>
                            <div className="flex flex-col gap-1">
                                <span className="text-xs text-gray-900 font-bold tracking-wider">
                                    SOURCE ORIGIN
                                </span>
                                <span className="font-medium text-gray-800">
                                    {value.sourceMethod === "file" ?
                                        value.sourceFile?.name || "--" :
                                        value.hfRepoId || "--"}
                                </span>
                            </div>
                            <div className="flex flex-col gap-1">
                                <span className="text-xs text-gray-900 font-bold tracking-wider">
                                    TRAINING SCHEMA
                                </span>
                                <span className="font-medium text-gray-800">
                                    {value.targetSchemaId || "--"}
                                </span>
                            </div>
                            <div className="flex flex-col gap-1">
                                <span className="text-xs text-gray-900 font-bold tracking-wider">
                                    RECORD COUNT
                                </span>
                                <span className="font-bold text-gray-500">
                                    {value.detectedSample || "--"} samples
                                </span>
                            </div>
                            <div className="flex flex-col gap-1">
                                <span className="text-xs text-gray-900 font-bold tracking-wider">
                                    INGESTED BY
                                </span>
                                <span className="font-medium text-gray-800">
                                    Maulana M.
                                </span>
                            </div>
                        </div>

                        <div className="flex flex-col gap-1.5 pt-2">
                            <span className="text-xs text-gray-900 font-bold tracking-wider">
                                MANAGED STORAGE URI (AUTO-ASSIGNED)
                            </span>
                            <div className="flex justify-between items-center bg-gray-500 text-gray-100 p-3 rounded-md font-mono text-xs">
                                <span>s3://defnex-mlops/datasets/{value.datasetId || "dataset"}/{value.datasetVersion || "v0.0.0"}/</span>
                                <span className="text-gray-400 font-sans text-xs">S3/GCS</span>
                            </div>
                        </div>
                    </div>
                </FieldGroup>
            </ScrollArea>
            <div className="flex justify-between items-center gap-2">
				<Button
					type="button"
					variant={"outline"}
					onClick={() => setStepActive(4)}>
					Back
				</Button>
				<Button
					type="button"
                    className="bg-emerald-600 text-white hover:bg-emerald-700"
					onClick={() => {
						submitDataset();
						setStepActive(6);
					}}>
					Create Immutable Dataset Version
				</Button>
			</div>
        </>
    );
}