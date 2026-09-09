import { Button } from "../ui/button";
import { Dialog as DialogPrimitive } from "@base-ui/react/dialog"
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "../ui/dialog";
import { Field, FieldGroup } from "../ui/field";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Textarea } from "../ui/textarea";
import { SlidersVertical, XIcon } from "lucide-react";
import { baseModels, datasets } from "~/utils";

export default function ModalRunSFT({ isOpen, toggleModal }: { isOpen: boolean, toggleModal: () => void }) {
    return (
        <Dialog open={isOpen}>
            <form>
                <DialogContent className="sm:max-w-lg" showCloseButton={false}>
                    <DialogHeader>
                        <DialogTitle>Launch SFT Training Run</DialogTitle>
                        <DialogDescription>
                        Queue an SFT fine-tuning job with Unsloth configuration
                        </DialogDescription>
                    </DialogHeader>
                    <FieldGroup>
                        <Field>
                            <Label htmlFor="target-model-id">Target Model ID</Label>
                            <Input id="target-model-id" name="target-model-id" placeholder="Enter target model ID" />
                        </Field>
                        <Field className="w-full">
                            <Label htmlFor="version-tag" aria-required>Dataset Format</Label>
                            <Select items={baseModels.map(f => ({ label: f.label, value: f.value }))} defaultValue={baseModels[0].value}>
                                <SelectTrigger className="w-full">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectGroup>
                                    {baseModels.map((item) => (
                                        <SelectItem key={item.value} value={item.value}>
                                        {item.label}
                                        </SelectItem>
                                    ))}
                                    </SelectGroup>
                                </SelectContent>
                            </Select>
                        </Field>
                        <div className="grid grid-cols-2 gap-4">
                            <Field className="w-full">
                                <Label htmlFor="version-tag" aria-required>Dataset Format</Label>
                                <Select items={datasets.map(f => ({ label: f.title, value: f.fileName }))} defaultValue={datasets[0].fileName}>
                                    <SelectTrigger className="w-full">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectGroup>
                                        {datasets.map((item) => (
                                            <SelectItem key={item.fileName} value={item.fileName}>
                                            {item.title}
                                            </SelectItem>
                                        ))}
                                        </SelectGroup>
                                    </SelectContent>
                                </Select>
                            </Field>
                            <Field>
                                <Label htmlFor="dataset-version">Dataset Version</Label>
                                <Input id="dataset-version" name="dataset-version" placeholder="Enter dataset version" />
                            </Field>
                        </div>
                        <div className="flex flex-col gap-2 p-4 border rounded-md">
                            <div>
                                <h3 className="text-sm font-bold flex items-center gap-2"><SlidersVertical className="w-4 h-4" />Hyperparameters (TrainingConfig)</h3>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                <div className="flex flex-col gap-2">
                                    <Field className="gap-1">
                                        <Label htmlFor="learning-rate" className="text-xs">Learning Rate</Label>
                                        <Input id="learning-rate" name="learning-rate" placeholder="Enter learning rate" />
                                    </Field>
                                    <Field className="gap-1">
                                        <Label htmlFor="epochs" className="text-xs">Epochs</Label>
                                        <Input id="epochs" name="epochs" placeholder="Enter number of epochs" />
                                    </Field>
                                </div>
                                <div className="flex flex-col gap-2">
                                    <Field className="gap-1">
                                        <Label htmlFor="batch-size" className="text-xs">Batch Size</Label>
                                        <Input id="batch-size" name="batch-size" placeholder="Enter batch size" />
                                    </Field>
                                    <Field className="gap-1">
                                        <Label htmlFor="lora-rank" className="text-xs">LoRA Rank (r)</Label>
                                        <Input id="lora-rank" name="lora-rank" placeholder="Enter LoRA rank" />
                                    </Field>
                                </div>
                            </div>
                        </div>
                        <Field>
                            <Label htmlFor="triggered-by" aria-required>Triggered By</Label>
                            <Input id="triggered-by" name="triggered-by" placeholder="Enter your name" required />
                        </Field>
                    </FieldGroup>
                    <DialogFooter>
                        <DialogClose onClick={toggleModal} render={<Button variant="outline">Cancel</Button>} />
                        <Button type="submit">Queue SFT Training Run</Button>
                    </DialogFooter>
                    <DialogPrimitive.Close
                        onClick={toggleModal}
                        data-slot="dialog-close"
                        render={
                        <Button
                            variant="ghost"
                            className="absolute top-4 right-4 bg-secondary"
                            size="icon-sm"
                        />
                        }
                    >
                        <XIcon />
                        <span className="sr-only">Close</span>
                    </DialogPrimitive.Close>
                </DialogContent>
            </form>
        </Dialog>
    )
}