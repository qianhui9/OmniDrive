"""Expired computation or observation hints expand only affected dependencies."""
import pytest
import torch
from .test_source_linked_admission import cold,bundle_for
