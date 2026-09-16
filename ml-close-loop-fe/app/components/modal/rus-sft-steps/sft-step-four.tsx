import {
    Field,
    FieldDescription,
    FieldGroup,
    FieldLabel,
} from "../../ui/field";
import { Input } from "../../ui/input";
import { Button } from "../../ui/button";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "../../ui/select";
import { ScrollArea } from "~/components/ui/scroll-area";
import { SlidersHorizontal } from "lucide-react";
import type { LoraConfigPropsSft, TrainingParapPropsSft } from "../run-sft";

export type StepFourValueProps = {
    loraConf: LoraConfigPropsSft;
    trainingParams: TrainingParapPropsSft;
}

export function SftStepFour({value, onChange, setStepActive}: {value: StepFourValueProps, onChange: (patch: Partial<StepFourValueProps>) => void, setStepActive: (step: number) => void}) {
    return (
        <>
            <ScrollArea className="max-h-125">
                <FieldGroup>
                    <div className="flex flex-col gap-4">
                        {/* Bagian Lora Configuration */}
                        <div className="flex flex-col gap-3 border p-4 rounded-lg bg-white">
                            <div className="flex items-center gap-2">
                                <SlidersHorizontal className="h-4 w-4 text-muted-foreground" />
                                <span className="text-xs font-bold text-muted-foreground uppercase">
                                    Lora Configuration
                                </span>
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Rank
                                    </FieldLabel>
                                    <Input value={value.loraConf.rank} onChange={(e) => onChange({ loraConf: { ...value.loraConf, rank: Number(e.target.value) } })} />
                                </Field>
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Alpha
                                    </FieldLabel>
                                    <Input value={value.loraConf.alpha} onChange={(e) => onChange({ loraConf: { ...value.loraConf, alpha: Number(e.target.value) } })} />
                                </Field>
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Dropout
                                    </FieldLabel>
                                    <Input value={value.loraConf.dropout} onChange={(e) => onChange({ loraConf: { ...value.loraConf, dropout: Number(e.target.value) } })} />
                                </Field>
                            </div>
                            <Field>
                                <FieldLabel className="text-xs text-muted-foreground">
                                    Target Modules (comma separated)
                                </FieldLabel>
                                <Input
                                    defaultValue="q_proj, k_proj, v_proj, o_proj"
                                    readOnly
                                />
                            </Field>
                        </div>

                        {/* Bagian Training Parameters */}
                        <div className="flex flex-col gap-3 border p-4 rounded-lg bg-white">
                            <span className="text-xs font-bold text-muted-foreground uppercase">
                                Training Parameters
                            </span>
                            <div className="grid grid-cols-3 gap-3">
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Epochs
                                    </FieldLabel>
                                    <Input value={value.trainingParams.epochs} onChange={(e) => onChange({ trainingParams: { ...value.trainingParams, epochs: Number(e.target.value) } })} />
                                </Field>
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Learning Rate
                                    </FieldLabel>
                                    <Input value={value.trainingParams.learningRate} onChange={(e) => onChange({ trainingParams: { ...value.trainingParams, learningRate: Number(e.target.value) } })} />
                                </Field>
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Batch Size
                                    </FieldLabel>
                                    <Input value={value.trainingParams.batchSize} onChange={(e) => onChange({ trainingParams: { ...value.trainingParams, batchSize: Number(e.target.value) } })} />
                                </Field>
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Gradient Accum.
                                    </FieldLabel>
                                    <Input value={value.trainingParams.gradientAccumulationSteps} onChange={(e) => onChange({ trainingParams: { ...value.trainingParams, gradientAccumulationSteps: Number(e.target.value) } })} />
                                </Field>
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Warmup Ratio
                                    </FieldLabel>
                                    <Input value={value.trainingParams.warmupRatio} onChange={(e) => onChange({ trainingParams: { ...value.trainingParams, warmupRatio: Number(e.target.value) } })} />
                                </Field>
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Weight Decay
                                    </FieldLabel>
                                    <Input value={value.trainingParams.weightDecay} onChange={(e) => onChange({ trainingParams: { ...value.trainingParams, weightDecay: Number(e.target.value) } })} />
                                </Field>
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Max Seq Length
                                    </FieldLabel>
                                    <Input value={value.trainingParams.maxSeqLength} onChange={(e) => onChange({ trainingParams: { ...value.trainingParams, maxSeqLength: Number(e.target.value) } })} />
                                </Field>
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Optimizer
                                    </FieldLabel>
                                    <Select value={value.trainingParams.optimizer} onValueChange={(newVal) => onChange({ trainingParams: { ...value.trainingParams, optimizer: newVal ?? "" } })}>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select optimizer" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="AdamW">
                                                AdamW
                                            </SelectItem>
                                            <SelectItem value="AdamW8bit">
                                                AdamW 8bit
                                            </SelectItem>
                                            <SelectItem value="SGD">
                                                SGD
                                            </SelectItem>
                                        </SelectContent>
                                    </Select>
                                </Field>
                                <Field>
                                    <FieldLabel className="text-xs text-muted-foreground">
                                        Scheduler
                                    </FieldLabel>
                                    <Select value={value.trainingParams.scheduler} onValueChange={(newVal) => onChange({ trainingParams: { ...value.trainingParams, scheduler: newVal ?? "" } })}>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select scheduler" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="Cosine">
                                                Cosine
                                            </SelectItem>
                                            <SelectItem value="Linear">
                                                Linear
                                            </SelectItem>
                                            <SelectItem value="Constant">
                                                Constant
                                            </SelectItem>
                                        </SelectContent>
                                    </Select>
                                </Field>
                            </div>
                        </div>

                        {/* Bagian Training Strategy Footer */}
                        <div className="flex justify-between items-center p-3 rounded-lg border border-blue-100 bg-blue-50/20 text-xs">
                            <span className="font-semibold text-blue-900">
                                Training Strategy
                            </span>
                            <span className="text-blue-700 font-medium">
                                Supervised Fine-Tuning (SFT) • Unsloth
                            </span>
                        </div>

                        <FieldDescription className="text-xs text-muted-foreground px-1">
                            This is the intended execution configuration for the execution configuration — not a live training invocation.
                        </FieldDescription>
                    </div>
                </FieldGroup>
            </ScrollArea>

            <div className="flex justify-between items-center gap-2">
				<Button
					type="button"
					variant={"outline"}
					onClick={() => setStepActive(3)}>
					Back
				</Button>
				<Button
					type="button"
					// disabled={!isAvailabletoNextStep()}
					// className={`${isAvailabletoNextStep() ? "" : "opacity-50 cursor-not-allowed"}`}
					onClick={() => setStepActive(5)}>
					Next
				</Button>
			</div>
        </>
    );
}