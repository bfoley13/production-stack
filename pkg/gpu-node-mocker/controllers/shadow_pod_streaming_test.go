/*
Copyright 2026 The KAITO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"context"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestEnsureShadowPodOmitsStreamingDependencies(t *testing.T) {
	for _, mode := range []string{"sas", "direct-azure", "local", "s3", "gcs"} {
		t.Run(mode, func(t *testing.T) {
			original := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "custom-model-0", Namespace: "models", UID: "original-uid",
					Labels: map[string]string{
						InferenceSetCreatedByLabelKey: "custom-model",
						"azure.workload.identity/use": "true",
					},
				},
				Spec: corev1.PodSpec{
					NodeName:           "fake-custom-model",
					ServiceAccountName: "kaito-model-streamer",
					Containers: []corev1.Container{{
						Name:  "inference",
						Args:  []string{"--model", "$STREAM_MODEL_URI", "--served-model-name", "custom-model"},
						Ports: []corev1.ContainerPort{{ContainerPort: 5000}},
						Env: []corev1.EnvVar{
							{Name: "AZURE_CLIENT_ID", Value: "original-client-id"},
							{Name: "AZURE_STORAGE_ACCOUNT_NAME", Value: "original-storage"},
						},
					}},
					Volumes: []corev1.Volume{
						{Name: "streaming-sas", VolumeSource: corev1.VolumeSource{
							EmptyDir: &corev1.EmptyDirVolumeSource{},
						}},
						{Name: "fetch-sas-script", VolumeSource: corev1.VolumeSource{
							ConfigMap: &corev1.ConfigMapVolumeSource{
								LocalObjectReference: corev1.LocalObjectReference{Name: "original-fetch-sas"},
							},
						}},
						{Name: "azure-identity-token", VolumeSource: corev1.VolumeSource{
							Projected: &corev1.ProjectedVolumeSource{},
						}},
					},
				},
			}
			switch mode {
			case "sas":
				original.Spec.InitContainers = []corev1.Container{{
					Name: "fetch-sas", Image: "python:3.12-slim",
					VolumeMounts: []corev1.VolumeMount{
						{Name: "streaming-sas", MountPath: "/mnt/streaming"},
						{Name: "fetch-sas-script", MountPath: "/scripts"},
						{Name: "azure-identity-token", MountPath: "/var/run/secrets/azure/tokens"},
					},
				}}
			case "direct-azure":
				original.Spec.Containers[0].Args[1] = "az://weights/custom-model"
			case "local":
				original.Spec.Containers[0].Args[1] = "/opt/kaito/models/custom-model"
			case "s3":
				original.Spec.Containers[0].Args[1] = "s3://weights/custom-model"
			case "gcs":
				original.Spec.Containers[0].Args[1] = "gs://weights/custom-model"
			}
			cl := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(original).Build()
			before := original.DeepCopy()
			r := &ShadowPodReconciler{Client: cl, Config: testConfig()}
			ctx := context.Background()

			shadow, err := r.ensureShadowPod(ctx, original, "shadow-models-custom-model-0")
			if err != nil {
				t.Fatalf("ensureShadowPod: %v", err)
			}
			if !reflect.DeepEqual(original, before) {
				t.Error("original inference pod was modified")
			}
			if len(shadow.Spec.InitContainers) != 0 {
				t.Errorf("shadow pod must have no init containers, got %+v", shadow.Spec.InitContainers)
			}
			if shadow.Spec.ServiceAccountName != "" {
				t.Errorf("shadow pod must not assume streaming ServiceAccount %q", shadow.Spec.ServiceAccountName)
			}
			if _, ok := shadow.Labels["azure.workload.identity/use"]; ok {
				t.Error("shadow pod must not opt into workload identity")
			}
			if len(shadow.Spec.Volumes) != 1 || shadow.Spec.Volumes[0].Name != "config" {
				t.Errorf("shadow pod must mount only simulator config, got %+v", shadow.Spec.Volumes)
			}
			if len(shadow.Spec.Containers) != 1 {
				t.Fatalf("expected one simulator container, got %d", len(shadow.Spec.Containers))
			}
			sim := shadow.Spec.Containers[0]
			if sim.Name != "llm-d-inference-sim" || sim.Image != DefaultInferenceSimImage {
				t.Errorf("unexpected simulator: %s %s", sim.Name, sim.Image)
			}
			if sim.ReadinessProbe == nil || sim.ReadinessProbe.HTTPGet.Path != "/ready" {
				t.Error("simulator readiness probe must be retained")
			}
			if sim.Ports[0].ContainerPort != 5000 {
				t.Errorf("original serving port not preserved: %+v", sim.Ports)
			}
			for _, e := range sim.Env {
				if strings.HasPrefix(e.Name, "AZURE_") || strings.HasPrefix(e.Name, "STREAM_") {
					t.Errorf("streaming environment %q leaked into simulator", e.Name)
				}
			}
			terms := shadow.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
			if terms[0].MatchExpressions[0].Key != LabelFakeNode {
				t.Error("shadow pod must still exclude fake nodes")
			}
			configMaps := &corev1.ConfigMapList{}
			if err := cl.List(ctx, configMaps); err != nil {
				t.Fatalf("list configmaps: %v", err)
			}
			if len(configMaps.Items) != 1 {
				t.Fatalf("expected only simulator configmap, got %d", len(configMaps.Items))
			}
			config := configMaps.Items[0].Data["config.yaml"]
			if !strings.Contains(config, "custom-model") || !strings.Contains(config, "force-dummy-tokenizer: true") {
				t.Error("simulator must retain the served model name and dummy tokenizer")
			}
		})
	}
}

func TestFirstUnreadyContainerReason(t *testing.T) {
	pod := &corev1.Pod{Status: corev1.PodStatus{
		InitContainerStatuses: []corev1.ContainerStatus{{
			Name: "init",
			State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{
				Reason: "CrashLoopBackOff", Message: "init failed",
			}},
		}},
		ContainerStatuses: []corev1.ContainerStatus{{
			Name:  "llm-d-inference-sim",
			State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "PodInitializing"}},
		}},
	}}
	reason, message := firstUnreadyContainerReason(pod)
	if !strings.Contains(reason, "CrashLoopBackOff") || message != "init failed" {
		t.Errorf("reason=%q message=%q", reason, message)
	}
	if r, m := firstUnreadyContainerReason(&corev1.Pod{}); r != "" || m != "" {
		t.Errorf("healthy pod should report nothing, got %q/%q", r, m)
	}
}
