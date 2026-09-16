import {
    FieldGroup,
    FieldLabel,
} from "../../ui/field";
import { Button } from "../../ui/button";
import clsx from "clsx";
import { ScrollArea } from "~/components/ui/scroll-area";
import { ExternalLink, Zap, CircleCheck, HardDrive } from "lucide-react";
import { useState } from "react";
import { providers } from "~/utils";

export type StepThreeValueProps = {
    provider: string;
    computeRes: string;
}

export function SftStepThree({value, onChange, setStepActive}: {value: StepThreeValueProps, onChange: (patch: Partial<StepThreeValueProps>) => void, setStepActive: (step: number) => void}) {
    const [selectedProvider, setSelectedProvider] = useState(providers.find(provider => provider.name === value.provider) || providers[0]);
    const [selectedComputeRes, setSelectedComputeRes] = useState(providers[0].computeRes[0]);

    return (
        <>
            <ScrollArea className="max-h-125">
                <FieldGroup>
                    <div className="flex flex-col gap-4">
                        {/* Bagian Provider */}
                        <div className="flex flex-col gap-2">
                            <FieldLabel className="text-xs font-bold text-muted-foreground uppercase">
                                Provider
                            </FieldLabel>
                            <div className="grid grid-cols-3 gap-3">
                                {providers.slice(0, 3).map((provider, index) => (
                                    <div
                                        key={index}
                                        className={clsx(
                                            "flex flex-col gap-1 p-3 rounded-lg border cursor-pointer",
                                            selectedProvider?.name === provider.name
                                                ? "border-black bg-blue-50/20"
                                                : "border-gray-300 bg-white"
                                        )}
                                        onClick={() => {
                                            if (!provider.disabled) {
                                                setSelectedProvider(provider);
                                                onChange({ 
                                                    provider: provider.name,
                                                    computeRes: provider.computeRes[0].name
                                                });
                                                setSelectedComputeRes(provider.computeRes[0]);
                                            }
                                        }}
                                    >
                                        <span className="font-bold text-sm text-black">
                                            {provider.name}
                                        </span>
                                        <span className="text-xs text-muted-foreground">
                                            {provider.description}
                                        </span>
                                    </div>
                                ))}
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                                {providers.slice(3).map((provider, index) => (
                                    <div
                                        key={index}
                                        className="flex flex-col gap-1 p-3 rounded-lg border border-gray-200 bg-gray-50 opacity-60 cursor-not-allowed"
                                    >
                                        <span className="font-bold text-sm text-gray-500">
                                            {provider.name}
                                        </span>
                                        <span className="text-xs text-gray-400">
                                            {provider.description}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Bagian Colab Runner Banner */}
                        {selectedProvider?.id === "google-colab" && (
                            <div className="flex flex-col gap-2 p-4 rounded-lg border border-amber-300 bg-amber-50/40">
                                <div className="flex justify-between items-start gap-4">
                                    <div className="flex flex-col gap-1">
                                        <span className="text-xs font-bold text-amber-800 uppercase">
                                            Colab Runner — DEFNEX_MLOPS_COLAB.IPYNB
                                        </span>
                                        <span className="text-xs font-mono text-amber-900 break-all">
                                            https://colab.research.google.com/github/defnex/demo/blob/main/DEFNEX_MLOPS_COLAB...
                                        </span>
                                    </div>
                                    <Button className="bg-orange-600 hover:bg-orange-700 text-white text-xs gap-1.5 shrink-0">
                                        <ExternalLink className="h-3.5 w-3.5" />
                                        Open Colab
                                    </Button>
                                </div>
                                <span className="text-xs text-amber-700/80 pt-1">
                                    Configured centrally in Infrastructure → Colab. Opens notebook template in a new tab.
                                </span>
                            </div>
                        )}

                        {/* Bagian Compute Resource */}
                        <div className="flex flex-col gap-2">
                            <FieldLabel className="text-xs font-bold text-muted-foreground uppercase">
                                Compute Resource
                            </FieldLabel>
                            <div className="flex flex-col gap-2">
                                {selectedProvider?.computeRes.map((res, index) => (
                                    <div
                                        key={index}
                                        className={clsx(
                                            "flex justify-between items-center p-3 rounded-lg border cursor-pointer",
                                            selectedComputeRes?.name === res.name
                                                ? "border-black bg-blue-50/20"
                                                : "border-gray-300 bg-white",
                                            res.busy && "opacity-60 bg-gray-50"
                                        )}
                                        onClick={() => {
                                            if (!res.busy) {
                                                setSelectedComputeRes(res);
                                                onChange({ computeRes: res.name });
                                            }
                                        }}
                                    >
                                        <div className="flex items-center gap-2">
                                            {res.isAuto && (
                                                <Zap className="h-4 w-4 text-blue-600 shrink-0" />
                                            )}
                                            <div className="flex flex-col gap-0.5">
                                                <span className="font-bold text-sm text-black">
                                                    {res.name}
                                                </span>
                                                <span className="text-xs text-muted-foreground">
                                                    {res.specs}
                                                </span>
                                            </div>
                                        </div>
                                        {res.status && (
                                            <div
                                                className={clsx(
                                                    "inline-flex items-center gap-1 px-2.5 py-1 rounded-md border text-xs font-medium shrink-0",
                                                    res.status === "Available"
                                                        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                                                        : "border-gray-200 bg-gray-100 text-muted-foreground"
                                                )}
                                            >
                                                {res.status === "Available" ? (
                                                    <CircleCheck className="h-3.5 w-3.5" />
                                                ) : (
                                                    <HardDrive className="h-3.5 w-3.5" />
                                                )}
                                                {res.status}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Bagian Resolved Target */}
                        <div className="flex justify-between items-center p-3 rounded-lg border bg-white text-xs text-muted-foreground">
                            <span>Resolved target:</span>
                            <span className="font-semibold text-black">
                                {selectedComputeRes.specs}
                                {/* colab-account-a (NVIDIA A100, 40GB) */}
                            </span>
                        </div>
                    </div>
                </FieldGroup>
            </ScrollArea>

            <div className="flex justify-between items-center gap-2">
				<Button
					type="button"
					variant={"outline"}
					onClick={() => setStepActive(2)}>
					Back
				</Button>
				<Button
					type="button"
					onClick={() => setStepActive(4)}>
					Next
				</Button>
			</div>
        </>
    );
}