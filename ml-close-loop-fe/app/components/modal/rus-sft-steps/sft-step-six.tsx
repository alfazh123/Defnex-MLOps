import {
    FieldGroup,
} from "../../ui/field";
import { Button } from "../../ui/button";
import { ScrollArea } from "~/components/ui/scroll-area";

export type StepSixValueProps = {
    dataset: string;
    baseModel: string;
    compute: string;
    training: string;
    samplesValidation: string;
    resultingModelVersion: string;
    lora: string;
    evaluation: string;
}

export function SftStepSix({value, setStepActive}: {value: StepSixValueProps, setStepActive: (step: number) => void}) {
    return (
        <>
            <ScrollArea className="max-h-125">
                <FieldGroup>
                    <div className="flex flex-col gap-4">
                        {/* Ringkasan Review Kartu Utama */}
                        <div className="flex flex-col gap-4 p-5 rounded-xl border border-blue-200 bg-blue-50/10 shadow-sm">
                            <div className="grid grid-cols-2 gap-6">
                                {/* Kolom Kiri */}
                                <div className="flex flex-col gap-4">
                                    <div className="flex flex-col gap-1">
                                        <span className="text-xs font-bold text-muted-foreground uppercase">
                                            Dataset
                                        </span>
                                        <span className="text-sm font-semibold text-black">
                                            {value.dataset}
                                        </span>
                                    </div>
                                    <div className="flex flex-col gap-1">
                                        <span className="text-xs font-bold text-muted-foreground uppercase">
                                            Base Model
                                        </span>
                                        <span className="text-sm font-semibold text-black">
                                            {value.baseModel}
                                        </span>
                                    </div>
                                    <div className="flex flex-col gap-1">
                                        <span className="text-xs font-bold text-muted-foreground uppercase">
                                            Compute
                                        </span>
                                        <span className="text-sm font-semibold text-black">
                                            {value.compute}
                                        </span>
                                    </div>
                                    <div className="flex flex-col gap-1">
                                        <span className="text-xs font-bold text-muted-foreground uppercase">
                                            Training
                                        </span>
                                        <span className="text-sm font-semibold text-black">
                                            {value.training}
                                        </span>
                                    </div>
                                </div>

                                {/* Kolom Kanan */}
                                <div className="flex flex-col gap-4">
                                    <div className="flex flex-col gap-1">
                                        <span className="text-xs font-bold text-muted-foreground uppercase">
                                            Samples / Validation
                                        </span>
                                        <span className="text-sm font-semibold text-black">
                                            {value.samplesValidation}
                                        </span>
                                    </div>
                                    <div className="flex flex-col gap-1">
                                        <span className="text-xs font-bold text-muted-foreground uppercase">
                                            Resulting Model Version
                                        </span>
                                        <span className="text-sm font-semibold text-blue-600">
                                            {value.resultingModelVersion}
                                        </span>
                                    </div>
                                    <div className="flex flex-col gap-1">
                                        <span className="text-xs font-bold text-muted-foreground uppercase">
                                            Lora
                                        </span>
                                        <span className="text-sm font-semibold text-black font-mono">
                                            {value.lora}
                                        </span>
                                    </div>
                                    <div className="flex flex-col gap-1">
                                        <span className="text-xs font-bold text-muted-foreground uppercase">
                                            Evaluation
                                        </span>
                                        <span className="text-sm font-semibold text-black">
                                            {value.evaluation}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </FieldGroup>
            </ScrollArea>

            <div className="flex justify-between items-center gap-2">
				<Button
					type="button"
					variant={"outline"}
					onClick={() => setStepActive(5)}>
					Back
				</Button>
				<Button
					type="button"
					// disabled={!isAvailabletoNextStep()}
					// className={`${isAvailabletoNextStep() ? "" : "opacity-50 cursor-not-allowed"}`}
					onClick={() => setStepActive(7)}>
					Next
				</Button>
			</div>
        </>
    );
}