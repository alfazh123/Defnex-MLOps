// import { baseModels, datasets, signal1, signal2, signal3 } from "~/utils";
import {
	Dialog,
	DialogClose,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "../ui/dialog";
import { Field, FieldGroup } from "../ui/field";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Button } from "../ui/button";

import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import type { TrainingRun } from "~/type";
import { Separator } from "../ui/separator";
import { Textarea } from "../ui/textarea";
import { XIcon } from "@phosphor-icons/react/dist/ssr";

export default function ModalEvaluationModel({
	isOpen,
	toggleClose,
	modelRegist,
}: {
	isOpen: boolean;
	toggleClose: () => void;
	modelRegist: TrainingRun;
}) {
	return (
		<Dialog open={isOpen}>
			<form>
				<DialogContent
					className="sm:max-w-lg"
					showCloseButton={false}>
					<DialogHeader>
						<DialogTitle>
							Evaluation Center & 3 Core Signals
						</DialogTitle>
						<DialogDescription>
							model: <span>{modelRegist.datasetVersion}</span>
						</DialogDescription>
					</DialogHeader>
					<Separator />
					<FieldGroup>
						<div className="grid grid-cols-2 gap-4">
							{/* <Field>
                                <Label htmlFor="signal-1" className="font-semibold">Signal 1: Evaluation Loss Convergence Trend</Label>
                                <Select items={signal1.map(f => ({ label: f.label, value: f.value }))} id="signal-1">
                                    <SelectTrigger className="w-full">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectGroup>
                                        {signal1.map((item) => (
                                            <SelectItem key={item.value} value={item.value}>
                                            {item.label}
                                            </SelectItem>
                                        ))}
                                        </SelectGroup>
                                    </SelectContent>
                                </Select>
                            </Field> */}
							{/* <Field>
                                <Label htmlFor="signal-2">Signal 2: Qualitative Target Domain Quality</Label>
                                <Select items={signal2.map(f => ({ label: f.label, value: f.value }))} id="signal-2">
                                    <SelectTrigger className="w-full">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectGroup>
                                        {signal2.map((item) => (
                                            <SelectItem key={item.value} value={item.value}>
                                            {item.label}
                                            </SelectItem>
                                        ))}
                                        </SelectGroup>
                                    </SelectContent>
                                </Select>
                            </Field> */}
						</div>
						{/* <Field>
                            <Label htmlFor="signal-3">Signal 3: General Domain Regression Check</Label>
                            <Select items={signal3.map(f => ({ label: f.label, value: f.value }))} id="signal-3">
                                <SelectTrigger className="w-full">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectGroup>
                                    {signal3.map((item) => (
                                        <SelectItem key={item.value} value={item.value}>
                                        {item.label}
                                        </SelectItem>
                                    ))}
                                    </SelectGroup>
                                </SelectContent>
                            </Select>
                        </Field> */}
						<div className="grid grid-cols-2 gap-4">
							<Field>
								<Label htmlFor="evaluator-name">
									Evaluator Name
								</Label>
								<Input
									id="evaluator-name"
									name="evaluator-name"
									placeholder="Enter name"
								/>
							</Field>
							<Field>
								<Label htmlFor="holistic-quality-score">
									Holistic Quality Score (0 - 100)
								</Label>
								<Input
									id="holistic-quality-score"
									name="holistic-quality-score"
									placeholder="Enter score"
								/>
							</Field>
						</div>
						<Field>
							<Label htmlFor="evaluation-rationale">
								Evaluation Rationale & Benchmark Observations
							</Label>
							<Textarea
								id="evaluation-rationale"
								name="evaluation-rationale"
								placeholder="Enter rationale and observations"
							/>
						</Field>
					</FieldGroup>
					<DialogFooter>
						<DialogClose
							onClick={toggleClose}
							render={<Button variant="outline">Cancel</Button>}
						/>
						<Button type="submit">Submit Evaluation Record</Button>
					</DialogFooter>
					<DialogPrimitive.Close
						onClick={toggleClose}
						data-slot="dialog-close"
						render={
							<Button
								variant="ghost"
								className="absolute top-4 right-4 bg-secondary"
								size="icon-sm"
							/>
						}>
						<XIcon />
						<span className="sr-only">Close</span>
					</DialogPrimitive.Close>
				</DialogContent>
			</form>
		</Dialog>
	);
}
