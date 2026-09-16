import {
    Field,
    FieldDescription,
    FieldGroup,
    FieldLabel,
} from "../../ui/field";
import { Input } from "../../ui/input";
import { Button } from "../../ui/button";
import clsx from "clsx";
import { ScrollArea } from "~/components/ui/scroll-area";
import { useState, useEffect } from "react";
import { baseModelsSftModal } from "~/utils";

export type stepTwoValueProps = {
    baseModel: string;
    baseModelVersion: string;
}

export function SftStepTwo({value, onChange, setStepActive}: {value: stepTwoValueProps, onChange: (patch: Partial<stepTwoValueProps>) => void, setStepActive: (step: number) => void}) {
    const [selectedModel, setSelectedModel] = useState(baseModelsSftModal.find(model => model.name === value.baseModel) || baseModelsSftModal[0]);
    
    // Tambahkan state untuk form input agar bisa diubah dan disinkronkan
    const [modelFamily, setModelFamily] = useState(selectedModel.modelFamily);
    const [version, setVersion] = useState(selectedModel.version);

    // Sinkronkan state input ketika selectedModel berubah
    useEffect(() => {
        setModelFamily(selectedModel.modelFamily);
        setVersion(selectedModel.version);
    }, [selectedModel]);

    return (
        <>
            <ScrollArea className="max-h-125">
                <FieldGroup>
                    <div className="flex flex-col gap-4">
                        {/* Bagian Base Model */}
                        <div className="flex flex-col gap-2">
                            <FieldLabel className="text-xs font-bold text-muted-foreground uppercase">
                                Base Model
                            </FieldLabel>
                            <div className="flex flex-col gap-2">
                                {baseModelsSftModal.map((model, index) => (
                                    <div
                                        key={index}
                                        className={clsx(
                                            "flex justify-between items-center p-3 rounded-lg border cursor-pointer",
                                            model.name === selectedModel.name
                                                ? "border-black bg-blue-50/20"
                                                : "border-gray-300 bg-white",
                                            model.disabled &&
                                                "opacity-60 cursor-not-allowed bg-gray-50",
                                        )}
                                        onClick={() => {
                                            if (!model.disabled) {
                                                setSelectedModel(model);
                                                onChange({
                                                    baseModel: model.name,
                                                    baseModelVersion: model.version,
                                                });
                                            }
                                        }}
                                    >
                                        <div className="flex flex-col gap-1">
                                            <span className="font-bold text-sm text-black">
                                                {model.name}
                                            </span>
                                            <span className="text-xs text-muted-foreground">
                                                {model.details}
                                            </span>
                                        </div>
                                        <span
                                            className={clsx(
                                                "text-xs font-medium whitespace-nowrap",
                                                model.disabled
                                                    ? "text-muted-foreground bg-gray-100 px-2 py-1 rounded"
                                                    : "text-muted-foreground",
                                            )}
                                        >
                                            {model.vram}
                                        </span>
                                    </div>
                                ))}
                            </div>
                            <FieldDescription className="text-xs text-muted-foreground pt-1">
                                Resource availability depends on the selected
                                training compute.
                            </FieldDescription>
                        </div>

                        {/* Bagian Resulting Model Version */}
                        <div className="flex flex-col gap-3 border p-4 rounded-lg bg-white">
                            <FieldLabel className="text-xs font-bold text-muted-foreground uppercase">
                                Resulting Model Version
                            </FieldLabel>
                            <div className="grid grid-cols-2 gap-4">
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Model Family
                                    </FieldLabel>
                                    <Input
                                        value={modelFamily}
                                        onChange={(e) => setModelFamily(e.target.value)}
                                    />
                                </Field>
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Version
                                    </FieldLabel>
                                    <Input 
                                        value={version} 
                                        onChange={(e) => setVersion(e.target.value)}
                                    />
                                </Field>
                            </div>
                            <div className="flex justify-between items-center pt-2 text-xs text-muted-foreground border-t">
                                <span>Resulting identifier:</span>
                                <span className="font-semibold text-blue-600">
                                    {modelFamily}-{version}
                                </span>
                            </div>
                        </div>
                    </div>
                </FieldGroup>
            </ScrollArea>

            <div className="flex justify-between items-center gap-2">
				<Button
					type="button"
					variant={"outline"}
					onClick={() => setStepActive(1)}>
					Back
				</Button>
				<Button
					type="button"
					onClick={() => setStepActive(3)}>
					Next
				</Button>
			</div>
        </>
    );
}