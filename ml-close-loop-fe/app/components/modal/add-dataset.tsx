import { XIcon } from "lucide-react";
import { Button } from "../ui/button";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "../ui/dialog";
import { Field, FieldGroup } from "../ui/field";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Textarea } from "../ui/textarea";

import { Dialog as DialogPrimitive } from "@base-ui/react/dialog"

export default function AddDatasetModal({isOpen, toggleModal, datasetFormats}: {isOpen: boolean, toggleModal: () => void, datasetFormats: {id: string, name: string}[]}) {
    return (
        <Dialog open={isOpen}>
            <form>
                <DialogContent className="sm:max-w-lg" showCloseButton={false}>
                    <DialogHeader>
                        <DialogTitle>Upload New Knowledge Dataset</DialogTitle>
                        <DialogDescription>
                        Enter the details for your new knowledge item.
                        </DialogDescription>
                    </DialogHeader>
                    <FieldGroup>
                        <Field>
                        <Label htmlFor="dataset-id">Dataset ID</Label>
                        <Input id="dataset-id" name="dataset-id" placeholder="Enter dataset ID" />
                        </Field>
                        <div className="grid grid-cols-2 gap-4">
                            <Field>
                                <Label htmlFor="version-tag" aria-required>Version Tag</Label>
                                <Input id="version-tag" name="version-tag" placeholder="v1.2.0" required />
                            </Field>
                            <Field>
                                <Label htmlFor="total-samples" aria-required>Total Samples</Label>
                                <Input id="total-samples" name="total samples" placeholder="1200" required type="number" />
                            </Field>
                        </div>
                        <Field>
                            <Label htmlFor="storate-url" aria-required>Storage URL / Manifest Path</Label>
                            <Input id="storate-url" name="storage" placeholder="Enter storage URL or manifest path" required />
                        </Field>
                        <div className="grid grid-cols-2 gap-4">
                            <Field className="w-full">
                                <Label htmlFor="version-tag" aria-required>Dataset Format</Label>
                                <Select items={datasetFormats.map(f => ({ label: f.name, value: f.id }))} defaultValue={datasetFormats[0].id}>
                                    <SelectTrigger className="w-full">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectGroup>
                                        {datasetFormats.map((item) => (
                                            <SelectItem key={item.id} value={item.id}>
                                            {item.name}
                                            </SelectItem>
                                        ))}
                                        </SelectGroup>
                                    </SelectContent>
                                </Select>
                            </Field>
                            <Field>
                                <Label htmlFor="ingested-by" aria-required>Ingested By</Label>
                                <Input id="ingested-by" name="ingested by" placeholder="Enter name" required />
                            </Field>
                        </div>
                        <Field>
                            <Label htmlFor="description" aria-required>Inteake Note / Description</Label>
                            <Textarea id="description" name="description" placeholder="Enter description" required />
                        </Field>
                    </FieldGroup>
                    <DialogFooter>
                        <DialogClose onClick={toggleModal} render={<Button variant="outline">Cancel</Button>} />
                        <Button type="submit">Ingest Version Manifest</Button>
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
                        <XIcon
                        />
                        <span className="sr-only">Close</span>
                    </DialogPrimitive.Close>
                </DialogContent>
            </form>
        </Dialog>
    )
}