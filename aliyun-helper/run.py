#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from typing import Optional

from alibabacloud_ecs20140526.client import Client as Ecs20140526Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_ecs20140526 import models as ecs_20140526_models


def get_region_id() -> str:
    region = os.environ.get('ALIYUN_REGION_ID') or os.environ.get('ALIBABA_CLOUD_REGION_ID')
    return region or 'ap-northeast-2'


def create_client() -> Ecs20140526Client:
    """
    初始化 ECS Client（使用环境变量中的 AK）
    需要环境变量：
      - ALIBABA_CLOUD_ACCESS_KEY_ID
      - ALIBABA_CLOUD_ACCESS_KEY_SECRET
      - 可选：ALIYUN_REGION_ID（默认 ap-northeast-2）
    """
    region_id = get_region_id()
    config = open_api_models.Config(
	    type='access_key',
	    access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
	    access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
	)
    config.endpoint = f'ecs.{region_id}.aliyuncs.com'
    return Ecs20140526Client(config)


def create_from_template(launch_template_name: Optional[str] = None) -> None:
    client = create_client()
    region_id = get_region_id()
    template_name = launch_template_name or os.environ.get('ALIYUN_LAUNCH_TEMPLATE_NAME') or 'gpu'
    request = ecs_20140526_models.RunInstancesRequest(
        region_id=region_id,
        launch_template_name=template_name,
    )
    try:
        resp = client.run_instances(request)
        print('已发起按模板创建实例。')
        print(resp)
    except Exception as e:
        print(f'创建实例失败: {e}')


def view_instance_status() -> None:
    client = create_client()
    region_id = get_region_id()
    request = ecs_20140526_models.DescribeInstanceStatusRequest(region_id=region_id)
    try:
        resp = client.describe_instance_status(request)
        # 兼容 SDK 返回对象或 dict 的形式
        body = None
        if isinstance(resp, dict):
            body = resp.get('body')
        else:
            body = getattr(resp, 'body', resp)

        statuses = None
        if isinstance(body, dict):
            statuses = (
                body.get('InstanceStatuses', {})
                .get('InstanceStatus', [])
            )
        else:
            # Tea models: body.InstanceStatuses.InstanceStatus
            instance_statuses = getattr(body, 'instance_statuses', None) or getattr(body, 'InstanceStatuses', None)
            statuses = getattr(instance_statuses, 'instance_status', None) or getattr(instance_statuses, 'InstanceStatus', [])

        if not statuses:
            print('未查询到任何实例。')
            return

        print('InstanceId\tStatus')
        # 每个元素可能是 dict 或 TeaModel
        for s in statuses:
            if isinstance(s, dict):
                iid = s.get('InstanceId') or s.get('instance_id')
                st = s.get('Status') or s.get('status')
            else:
                iid = getattr(s, 'instance_id', None) or getattr(s, 'InstanceId', None)
                st = getattr(s, 'status', None) or getattr(s, 'Status', None)
            if iid and st:
                print(f'{iid}\t{st}')
            else:
                print(s)
    except Exception as e:
        print(f'查询实例状态失败: {e}')


def view_instances() -> None:
    """使用 DescribeInstances 接口列出实例并仅输出关键信息。"""
    client = create_client()
    region_id = get_region_id()
    request = ecs_20140526_models.DescribeInstancesRequest(region_id=region_id, page_size=100)
    try:
        resp = client.describe_instances(request)
        # 兼容 SDK 返回对象或 dict 的形式
        body = None
        if isinstance(resp, dict):
            body = resp.get('body')
        else:
            body = getattr(resp, 'body', resp)

        instances = []
        if isinstance(body, dict):
            instances = (
                body.get('Instances', {})
                .get('Instance', [])
            )
        else:
            # Tea models: body.Instances.Instance (list)
            instances_container = getattr(body, 'instances', None) or getattr(body, 'Instances', None)
            instances = getattr(instances_container, 'instance', None) or getattr(instances_container, 'Instance', [])

        if not instances:
            print('未查询到任何实例。')
            return

        print('InstanceId\tStatus\tInstanceType\tPublicIP\tPrivateIP\tZoneId')
        for ins in instances:
            if isinstance(ins, dict):
                iid = ins.get('InstanceId')
                st = ins.get('Status')
                itype = ins.get('InstanceType')
                zone = ins.get('ZoneId')
                # Public IP
                pub_list = []
                pub_obj = ins.get('PublicIpAddress') or {}
                pub_list = pub_obj.get('IpAddress') or []
                pub_ip = pub_list[0] if pub_list else ''
                # Private IP (prefer VpcAttributes.PrivateIpAddress.IpAddress)
                prv_list = []
                vpc_attr = ins.get('VpcAttributes') or {}
                prv_obj = (vpc_attr.get('PrivateIpAddress') or {})
                prv_list = prv_obj.get('IpAddress') or []
                if not prv_list:
                    # fallback to NetworkInterfaces[0].PrimaryIpAddress
                    nis = (ins.get('NetworkInterfaces') or {}).get('NetworkInterface') or []
                    if nis:
                        prv_ip_candidate = nis[0].get('PrimaryIpAddress')
                        prv_list = [prv_ip_candidate] if prv_ip_candidate else []
                prv_ip = prv_list[0] if prv_list else ''
            else:
                iid = getattr(ins, 'instance_id', None) or getattr(ins, 'InstanceId', None)
                st = getattr(ins, 'status', None) or getattr(ins, 'Status', None)
                itype = getattr(ins, 'instance_type', None) or getattr(ins, 'InstanceType', None)
                zone = getattr(ins, 'zone_id', None) or getattr(ins, 'ZoneId', None)
                # Public IP
                pub_container = getattr(ins, 'public_ip_address', None) or getattr(ins, 'PublicIpAddress', None)
                pub_ips = []
                if pub_container is not None:
                    pub_ips = getattr(pub_container, 'ip_address', None) or getattr(pub_container, 'IpAddress', None) or []
                pub_ip = pub_ips[0] if pub_ips else ''
                # Private IP
                prv_ip = ''
                vpc_attr = getattr(ins, 'vpc_attributes', None) or getattr(ins, 'VpcAttributes', None)
                if vpc_attr is not None:
                    prv_container = getattr(vpc_attr, 'private_ip_address', None) or getattr(vpc_attr, 'PrivateIpAddress', None)
                    if prv_container is not None:
                        prv_ips = getattr(prv_container, 'ip_address', None) or getattr(prv_container, 'IpAddress', None) or []
                        prv_ip = prv_ips[0] if prv_ips else ''
                if not prv_ip:
                    nis = getattr(ins, 'network_interfaces', None) or getattr(ins, 'NetworkInterfaces', None)
                    ni_list = getattr(nis, 'network_interface', None) or getattr(nis, 'NetworkInterface', None) or []
                    if ni_list:
                        prv_ip_candidate = getattr(ni_list[0], 'primary_ip_address', None) or getattr(ni_list[0], 'PrimaryIpAddress', None)
                        prv_ip = prv_ip_candidate or ''

            print(f'{iid}\t{st}\t{itype}\t{pub_ip}\t{prv_ip}\t{zone}')
    except Exception as e:
        print(f'查询实例失败: {e}')


def stop_instance(instance_id: str) -> None:
    client = create_client()
    request = ecs_20140526_models.StopInstanceRequest(
        instance_id=instance_id,
        stopped_mode='StopCharging',
    )
    try:
        resp = client.stop_instance(request)
        print(f'已发送停止实例请求: {instance_id}')
        print(resp)
    except Exception as e:
        print(f'停止实例失败: {e}')


def start_instance(instance_id: str) -> None:
    client = create_client()
    request = ecs_20140526_models.StartInstanceRequest(
        instance_id=instance_id,
    )
    try:
        resp = client.start_instance(request)
        print(f'已发送启动实例请求: {instance_id}')
        print(resp)
    except Exception as e:
        print(f'启动实例失败: {e}')


def main() -> None:
    print('请选择要执行的操作:')
    print('1) 按照模板创建')
    print('2) 查看实例列表/状态')
    print('3) 启动实例 (需要输入实例 ID)')
    print('4) 停止实例 (需要输入实例 ID)')
    choice = input('输入选项编号并回车: ').strip()

    if choice == '1':
        tpl = os.environ.get('ALIYUN_LAUNCH_TEMPLATE_NAME')
        if not tpl:
            tpl = input('请输入 Launch Template 名称 (默认: gpu): ').strip() or 'gpu'
        create_from_template(launch_template_name=tpl)
    elif choice == '2':
        # 使用 DescribeInstances API
        view_instances()
    elif choice == '3':
        instance_id = input('请输入要启动的实例 ID: ').strip()
        if not instance_id:
            print('实例 ID 不能为空。')
            sys.exit(1)
        start_instance(instance_id)
    elif choice == '4':
        instance_id = input('请输入要停止的实例 ID: ').strip()
        if not instance_id:
            print('实例 ID 不能为空。')
            sys.exit(1)
        stop_instance(instance_id)
    else:
        print('无效选项。')
        sys.exit(1)


if __name__ == '__main__':
    main()

