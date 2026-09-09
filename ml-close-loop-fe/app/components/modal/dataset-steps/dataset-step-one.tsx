import { useState } from "react";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "../../ui/field";
import { Label } from "../../ui/label";
import { Input } from "../../ui/input";
import { FileBox, FileText, Upload } from "lucide-react";
import { Separator } from "../../ui/separator";
import { Button } from "../../ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../ui/select";

export function DatasetStepOne() {
    const methods = [
        { label: "Upload File", description: "Local JSONL, CSV, or Excel files parsed & normalized", value: "file" },
        { label: "Hugging Face Dataset", description: "Direct import from public or private Hugging Face hub repository", value: "hf" },
    ]

    const [activeMethod, setActiveMethod] = useState("file");

    return (
        <>
            <FieldGroup>
                <h3>Select Dataset Source Type</h3>
                <div className="grid grid-cols-2 gap-4">
                    {methods.map((method) => (
                        <Field key={method.value} className={`flex flex-col gap-1 p-2 rounded-lg border ${activeMethod === method.value ? "border-black" : "border-gray-300 text-gray-300"}`} onClick={() => setActiveMethod(method.value)}>
                            <FieldLabel>
                                {method.label}
                            </FieldLabel>
                            <FieldDescription>
                                {method.description}
                            </FieldDescription>
                        </Field>
                    ))}

                    { activeMethod === "file" ? (
                        <div className="flex flex-col gap-2 p-2 rounded-lg w-full col-span-2">
                            <Label htmlFor="file-upload-input" className="flex flex-col h-52 gap-2 border-dashed w-full items-center justify-center border-2 border-gray-300 rounded-lg cursor-pointer hover:bg-gray-100">
                                <div className="flex flex-col gap-1 items-center justify-center bg-gray-200 p-2 rounded-full">
                                    <Upload className="mx-auto h-6 w-6 text-black" />
                                </div>
                                <p className="text-base font-semibold">Click to browse or drag and drop dataset file</p>
                                <span className="text-xs text-muted-foreground">Supported: JSONL, CSV, Excel (.xlsx)</span>
                            </Label>
                            <Input id="file-upload-input" type="file" accept=".jsonl,.csv,.xlsx" className="hidden" />

                            <div className="flex gap-1 p-2 rounded-lg border border-yellow-300 text-yellow-600 bg-amber-50 text-xs">
                                <FileText className="h-8 w-8 text-yellow-600" />
                                <p className="font-light"><span className="font-semibold">Format Normalization:</span> Non-JSONL files (such as CSV or Excel) will be automatically converted and validated into canonical JSONL ShareGPT format for training pipelines.</p>
                            </div>
                        </div>
                    ) : (
                        <div className="flex flex-col gap-2 p-2 rounded-lg w-full col-span-2">
                            <Field>
                                <FieldLabel htmlFor="hf-dataset-input">Hugging Face Hub Repository ID</FieldLabel>
                                <div className="flex gap-2 ">
                                    <Input id="hf-dataset-input" placeholder="e.g. username/dataset-name" />
                                    <Button>Validate</Button>
                                </div>
                                <FieldDescription className="text-xs">
                                    Examples: <span className="text-black">defnex/defense-scenarios-v1, Open-Orca/OpenOrca, tatsu-lab/alpaca</span>
                                </FieldDescription>
                            </Field>
                            <div className="grid grid-cols-2 gap-2">
                                <Field>
                                    <FieldLabel htmlFor="dataset-split">Dataset Split</FieldLabel>
                                    <Select id="dataset-split" defaultValue={"train"}>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select a split" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="train">Train</SelectItem>
                                            <SelectItem value="test">Test</SelectItem>
                                            <SelectItem value="validation">Validation</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </Field>
                                <Field>
                                    <FieldLabel htmlFor="revision">Revision / Git Branch</FieldLabel>
                                    <Input id="revision" placeholder="Enter revision or branch name" />
                                </Field>
                            </div>
                        </div>
                    )}
                    <div className="col-span-2">
                        <div className="flex flex-col gap-4 border p-2 rounded-lg">
                            <div className="flex justify-between items-center">
                                <div className="flex gap-1 items-center">
                                    <FileBox className="h-4 w-4 text-accent-foreground" />
                                    <p className="text-base">defnex_defense_scenarios_train.jsonl</p>
                                </div>
                                <span className="text-sm text-muted-foreground font-semibold">12.8 MB</span>
                            </div>
                            <Separator />
                            <div className="grid md:grid-cols-3 grid-cols-1 gap-2">
                                <div className="p-2 rounded-md border">
                                    <p className="font-semibold text-muted-foreground">Detected Sample</p>
                                    <span className="font-bold">15,240</span>
                                </div>
                                <div className="p-2 rounded-md border">
                                    <p className="font-semibold text-muted-foreground">Format Normalized</p>
                                    <span className="font-bold">JSONL</span>
                                </div>
                                <div className="p-2 rounded-md border">
                                    <p className="font-semibold text-muted-foreground">Checksum (SHA-256)</p>
                                    <span className="font-bold">6d2258a41920da...</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </FieldGroup>
        </>
    )
}