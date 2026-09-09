import { useState } from "react";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "../../ui/field";
import { Label } from "../../ui/label";
import { Input } from "../../ui/input";
import { CircleCheck, FileBox, FileText, Upload } from "lucide-react";
import { Separator } from "../../ui/separator";
import { Button } from "../../ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../ui/select";
import { datasets } from "~/utils";
import { Textarea } from "~/components/ui/textarea";

export function DatasetStepTwo() {
    const datasetFormatedList = datasets.map((data, id) => ({
        label: data.title,
        value: 'v' + (id + 1) + '.0.0',
    }))

    const methods = [
        { label: `Add Version to Existing Dataset (${datasets.length})`, value: "add" },
        { label: "Create New Dataset Catalog", value: "create" },
    ]

    const [activeMethod, setActiveMethod] = useState("add");
    const [versionSelected, setVersionSelected] = useState<string>(datasetFormatedList[0]?.value ?? "");

    const handleVersionChange = (value: string | null) => {
        setVersionSelected(value || "");
    }

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
                            {/* <FieldDescription>
                                {method.description}
                            </FieldDescription> */}
                        </Field>
                    ))}

                    { activeMethod === "add" ? (
                        <div className="flex flex-col gap-2 p-2 rounded-lg w-full col-span-2">
                            <Field>
                                <Label>Choose Dataset</Label>
                                <Select items={datasetFormatedList} onValueChange={(value) => handleVersionChange(value)} value={versionSelected}>
                                    <SelectTrigger>
                                        <SelectValue placeholder="Select Dataset" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {datasetFormatedList.map((item) => (
                                            <SelectItem key={item.value} value={item.value}>
                                                {item.label}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </Field>
                            <div className="flex flex-col p-2 bg-gray-600/10 rounded-md">
                                <div className="flex justify-between">
                                    <p>Current latest version:</p>
                                    <span>v1.0.0</span>
                                </div>
                                <div className="flex justify-between">
                                    <p>Suggested next increment:</p>
                                    <span>{versionSelected}</span>
                                </div>
                            </div>

                            <Field>
                                <Label>Target Version Tag <span className="text-red-500">*</span></Label>
                                <div className="relative">
                                    <Input placeholder="e.g. v1.2.0" value={versionSelected} className="h-10" />
                                    <div className="absolute right-3 top-2.5 flex gap-1 items-center text-green-400 border border-green-500 bg-green-50 px-1 rounded-sm font-semibold"><CircleCheck className="text-green h-4 w-4" /><span>Imutable</span></div>
                                </div>
                                <FieldDescription>Dataset versions are immutable. Once created, version v1.2.0 cannot be modified or replaced.</FieldDescription>
                            </Field>
                        </div>
                    ) : (
                        <div className="grid md:grid-cols-2 grid-cols-1 gap-2 p-2 rounded-lg w-full col-span-2">
                            <Field>
                                <Label>New Dataset ID <span className="text-red-500">*</span></Label>
                                <Input placeholder="e.g. ds-defense-screnarios" />
                            </Field>
                            <Field>
                                <Label>Dataset Display Name <span className="text-red-500">*</span></Label>
                                <Input placeholder="e.g. Defense Scenarios Corpus" />
                            </Field>
                            <Field className="md:col-span-2">
                                <Label>Target Version Tag <span className="text-red-500">*</span></Label>
                                <div className="relative">
                                    <Input placeholder="e.g. v1.2.0" value={versionSelected} className="h-10" />
                                    <div className="absolute right-3 top-2.5 flex gap-1 items-center text-green-400 border border-green-500 bg-green-50 px-1 rounded-sm font-semibold"><CircleCheck className="text-green h-4 w-4" /><span>Imutable</span></div>
                                </div>
                                <FieldDescription>Dataset versions are immutable. Once created, version v1.2.0 cannot be modified or replaced.</FieldDescription>
                            </Field>
                        </div>
                    )}
                    <div className="col-span-2">
                        <div className="flex flex-col gap-4 p-2 rounded-lg">
                            <Field>
                                <Label>Ingested By (Authenticated Operator)</Label>
                                <div className="flex flex-wrap justify-between items-center gap-2 bg-gray-600/10 p-2 rounded-md">
                                    <div className="flex items-center gap-1 text-xs">
                                        <div className="rounded-full bg-amber-200 p-1 w-8 h-8 flex justify-center items-center">M</div>
                                        <p>Maulana M.</p>
                                        <span>(ML_ENGINEER)</span>
                                    </div>
                                    <p className="text-slate-500 font-semibold">READ-ONLY SYSTEM CONTEXT</p>
                                </div>
                            </Field>
                            <Field>
                                <Label htmlFor="intake-notes">Intake Notes / Description (Optional)</Label>
                                <Textarea id="intake-notes" placeholder="Enter intake notes..." />
                            </Field>
                        </div>
                    </div>
                </div>
            </FieldGroup>
        </>
    )
}