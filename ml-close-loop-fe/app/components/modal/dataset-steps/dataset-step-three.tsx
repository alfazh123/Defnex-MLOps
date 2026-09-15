import { FieldGroup, FieldLabel } from "~/components/ui/field";
import { RadioGroup, RadioGroupItem } from "~/components/ui/radio-group";
import { targetSchemas } from "~/utils";
// import { Controller } from "react-hook-form";
import { Button } from "~/components/ui/button";
import { ScrollArea } from "~/components/ui/scroll-area";

export type StepThreeValueProps = {
    targetSchemaId: string;
}

export function DatasetStepThree({
    value,
    onChange,
    setStepActive,
}: {
    value: StepThreeValueProps;
    onChange: (patch: Partial<StepThreeValueProps>) => void;
    setStepActive: (step: number) => void;
}) {

    return (
        <>
            <ScrollArea className="max-h-125">
                <FieldGroup>
                    <h3>Target Training Schema</h3>
                            <RadioGroup
                                className="grid grid-cols-2 gap-3"
                                value={value.targetSchemaId || targetSchemas[0].id}
                                onValueChange={(value) => {
                                    // onChange(value ?? "")
                                    onChange({ targetSchemaId: value ?? "" })
                                }}
                                // defaultValue={value.targetSchemaId[0] ?? ""}
                            >
                                {targetSchemas.map((schema) => (
                                    <div
                                        key={schema.id}
                                        className="flex gap-3 p-4 rounded-lg border transition-all cursor-pointer has-checked:border-primary has-checked:bg-primary/5"
                                    >
                                        <RadioGroupItem value={schema.id} id={schema.id} className="sr-only hidden" />

                                        <FieldLabel htmlFor={schema.id} className="cursor-pointer w-full flex flex-col items-start">
                                            <div className="flex justify-between items-center w-full">
                                                <p className="font-semibold">{schema.title}</p>
                                                <span className="text-xs text-muted-foreground bg-muted-foreground/10 px-2 py-1 rounded-sm">{schema.version}</span>
                                            </div>
                                            <span className="text-xs text-muted-foreground">{schema.description}</span>
                                        </FieldLabel>
                                    </div>
                                ))}
                            </RadioGroup>
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
    )
}