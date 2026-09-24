import { useState } from "react";
import { useNavigate } from "react-router";
import InputSection from "~/components/Input-section";
import { Button } from "~/components/ui/button";
import { Field, FieldGroup, FieldSet } from "~/components/ui/field";
import { Separator } from "~/components/ui/separator";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();

		if (username !== "" || password !== "") {
			sessionStorage.setItem("username", username);
			sessionStorage.setItem("password", password);

			navigate("/");
		}
  };

  return (
		<div className="flex w-full min-h-screen bg-[#f8f9fa] justify-center items-center p-4 font-sans text-slate-800">
			<div className="grid md:grid-cols-2 bg-white border border-slate-200/80 rounded-[32px] w-full max-w-4xl min-h-[500px] shadow-sm overflow-hidden p-3 gap-2">
				{/* Banner Kiri (Gaya Bento Modern) */}
				<div className="flex flex-col justify-between p-8 bg-slate-900 rounded-[24px] text-white relative overflow-hidden min-h-[280px]">
					{/* Subtle Background Accent Glow */}
					<div className="absolute -top-12 -left-12 w-48 h-64 bg-blue-500 rounded-full blur-3xl pointer-events-none" />
					<div className="absolute -bottom-12 -right-12 w-48 h-48 bg-emerald-500 rounded-full blur-3xl pointer-events-none" />

					{/* Logo Brand / Icon */}
					<div className="flex items-center gap-3 z-10">
						<div className="w-9 h-9 rounded-full bg-white text-slate-900 flex items-center justify-center font-bold text-base shadow-sm">
							D
						</div>
						<span className="text-lg font-bold tracking-tight italic">
							Defnex
						</span>
					</div>

					{/* Welcome Text */}
					<div className="space-y-2 z-10 mt-auto">
						<h1 className="text-2xl font-bold tracking-tight leading-snug">
							Welcome to ML Closed Loop
						</h1>
						<p className="text-xs text-slate-400 leading-relaxed">
							Start fine-tuning your model with your own dataset.
							Please log in to continue.
						</p>
					</div>
				</div>

				{/* Form Kanan */}
				<div className="flex flex-col justify-center w-full max-w-md mx-auto p-6 md:p-8 gap-6">
					<div className="flex flex-col gap-1">
						<h2 className="text-2xl font-bold tracking-tight text-slate-900">
							Get Started
						</h2>
						<p className="text-xs text-slate-500">
							Make your account and start fine-tuning your model with your own dataset.
						</p>
					</div>

					<Separator className="bg-slate-100" />

					<form
						onSubmit={handleSubmit}
						className="w-full">
						<FieldGroup className="space-y-4">
							<FieldSet className="space-y-3">
								<FieldGroup className="space-y-3">
									<InputSection
										id="username"
										label="Username"
										type="text"
										onChange={(e) =>
											setUsername(e.target.value)
										}
										required={true}
										className="h-11 bg-slate-50/80 border-slate-200 text-sm"
									/>
									<InputSection
										id="password"
										label="Password"
										type="password"
										onChange={(e) =>
											setPassword(e.target.value)
										}
										required={true}
										className="h-11 bg-slate-50/80 border-slate-200 text-sm"
									/>
									<InputSection
										id="password"
										label="Confirmation Password"
										type="password"
										onChange={(e) =>
											setPassword(e.target.value)
										}
										required={true}
										className="h-11 bg-slate-50/80 border-slate-200 text-sm"
									/>
								</FieldGroup>
							</FieldSet>

							<Field className="pt-2">
								<Button
									type="submit"
									className="w-full h-11 text-white font-semibold text-sm transition-all shadow-xs">
									Login
								</Button>
							</Field>
						</FieldGroup>
					</form>

					<p className="text-xs text-slate-500 text-center">
						Already have an account?{" "}
						<a
							href="/auth/login"
							className="text-sky-600 font-semibold hover:underline">
							Login
						</a>
					</p>
				</div>
			</div>
		</div>
  );
}