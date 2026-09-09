import { Separator } from "./ui/separator";
import { Button } from "./ui/button";

import type { TrainingRun } from "~/type";
import clsx from "clsx";

import { ArrowRight, Copy } from "lucide-react";
import { toast } from "./ui/toast";
import { useNavigate } from "react-router";

export default function DetailTrainRun({run}: {run: TrainingRun}) {
    const navigate = useNavigate();

    const handleSubmit = () => {
        navigate(`/model-registry`);
    }

    return (
        <div key={run.runId} className="bg-white border rounded-md p-4">
            <div className="flex flex-wrap justify-between items-center font-semibold text-sm mb-2">
                <div>
                    <h3 className="text-sm font-semibold">Run Details</h3>
                    <p className="text-xs text-accent-foreground font-light">{run.runId}</p>
                </div>
                <div>
                    <span className={clsx(
                        "px-2 py-1 rounded-md text-xs font-medium",
                        run.status === "COMPLETED" &&
                            "bg-green-100 text-green-800",
                        run.status === "RUNNING" &&
                            "bg-blue-100 text-blue-800",
                        run.status === "PENDING" &&
                            "bg-yellow-100 text-yellow-800",
                        run.status === "FAILED" &&
                            "bg-red-100 text-red-800",
                    )}>
                        {run.status}
                    </span>
                </div>
            </div>
            <Separator />
            <div className="flex flex-col space-y-4 mt-4">
                <div className="grid md:grid-cols-2 grid-cols-1 gap-4">
                    <div className="flex flex-col border p-2 rounded-md gap-2 border-blue-200 bg-blue-50">
                        <p className="text-blue-800 font-semibold">Final Eval Loss</p>
                        <h3 className="font-extrabold text-xl">{run.evalLoss ? run.evalLoss.toFixed(2) : "-"}</h3>
                    </div>
                    <div className="flex flex-col border p-2 rounded-md gap-2">
                        <p className="font-semibold">Train Loss</p>
                        <h3 className="font-extrabold text-xl">{run.trainLoss ? run.trainLoss.toFixed(2) : "-"}</h3>
                    </div>
                </div>
                <div className="flex flex-col gap-2">
                    <h3 className="font-bold text-base">Training Config (LoRA)</h3>
                    <div className="grid grid-cols-2 bg-accent border border-gray-200 rounded-md p-2 text-sm">
                        <div className="flex flex-col gap-2">
                            <p>LR: <span className="font-bold">{run.loraConfig.lr}</span></p>
                            <p>Epochs: <span className="font-bold">{run.loraConfig.epochs}</span></p>
                            <p>Max Seq: <span className="font-bold">{run.loraConfig.maxSeqLength} tokens</span></p>
                        </div>
                        <div className="flex flex-col gap-2">
                            <p>Batch: <span className="font-bold">{run.loraConfig.batch}</span></p>
                            <p>LoRA r/α: <span className="font-bold">{run.loraConfig.loraRank}/{run.loraConfig.loraAlpha}</span></p>
                        </div>
                    </div>
                </div>
                <div className="flex flex-col gap-2">
                    <h3 className="font-bold text-base">Checkpoint Artifact URI</h3>
                    <div id={`checkppoint-${run.runId}`} className="relative bg-accent border border-gray-200 rounded-md p-2 text-sm break-all">
                        <div className="absolute top-2 right-2">
                            <Copy className="h-4 w-4 text-accent-foreground cursor-pointer"
                            onClick={() => {
                                const checkpointPath = run.checkpointPath;
                                navigator.clipboard.writeText(checkpointPath);
                                toast.add({
                                    title: "Copied to clipboard",
                                    description: "Checkpoint path copied to clipboard",
                                    type: "success",
                                });
                            }} />
                        </div>
                        {run.checkpointPath}
                    </div>
                </div>
                <div className="flex justify-center items-center w-full">
                    <Button className="w-full" variant={"default"} onClick={handleSubmit}>
                        View Registered Model in Registry <ArrowRight />
                    </Button>
                </div>
            </div>
        </div>
    )
}